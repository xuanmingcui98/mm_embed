import os
import argparse
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors


@torch.inference_mode()
def compute_topk_cosine(
    queries: torch.Tensor,   # [N, D]
    targets: torch.Tensor,   # [N, D]
    k: int = 100,
    q_chunk: int = 4096,
    t_chunk: int = 65536,
    device: str = None,
    use_fp16: bool = True,
):
    """
    Exact cosine-similarity top-k for each query over all targets.
    NOTE: self (same row index) is NOT excluded, so the GT target
    can appear in the top-k list.

    Returns:
        top_indices: LongTensor [N, k]  -- indices in [0, N)
        top_scores:  FloatTensor [N, k] -- cosine similarity scores
    """
    assert queries.ndim == 2 and targets.ndim == 2
    assert queries.shape == targets.shape
    N, D = queries.shape
    k = min(k, N)  # we allow self, so up to N

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize for cosine similarity
    queries = F.normalize(queries.float(), dim=-1, eps=1e-8).contiguous()
    targets = F.normalize(targets.float(), dim=-1, eps=1e-8).contiguous()

    # We’ll keep final top-k on CPU
    top_scores_all = torch.full((N, k), -1e9, dtype=torch.float32)
    top_index_all = torch.full((N, k), -1, dtype=torch.long)

    # Cache targets on CPU, stream to device in blocks
    targets_cpu = targets.to("cpu", non_blocking=True)

    dev_dtype = torch.float16 if (use_fp16 and device.startswith("cuda")) else torch.float32

    for qs in tqdm(range(0, N, q_chunk)):
        qe = min(qs + q_chunk, N)
        qB = qe - qs

        # Query chunk
        q = queries[qs:qe].to(device, non_blocking=True).to(dev_dtype)  # [qB, D]

        # Running top-k for this query block
        chunk_scores = torch.full((qB, k), -1e9, device=device, dtype=torch.float32)
        chunk_index = torch.full((qB, k), -1, device=device, dtype=torch.long)

        for ts in range(0, N, t_chunk):
            te = min(ts + t_chunk, N)
            t = targets_cpu[ts:te].to(device, non_blocking=True).to(dev_dtype)  # [tB, D]
            tB = t.size(0)

            # Cosine similarity via inner product, both normalized
            scores = (q @ t.transpose(0, 1)).to(torch.float32)  # [qB, tB]

            # Local top-k within this block (includes self if in this block)
            k_local = min(k, tB)
            blk_scores, blk_idx = torch.topk(scores, k=k_local, dim=1)  # [qB, k_local]
            blk_idx = blk_idx + ts  # map to global indices

            # Merge with running block’s top-k
            merged_scores = torch.cat([chunk_scores, blk_scores], dim=1)  # [qB, k + k_local]
            merged_index = torch.cat([chunk_index, blk_idx], dim=1)       # [qB, k + k_local]

            new_scores, new_pos = torch.topk(merged_scores, k=k, dim=1)   # [qB, k]
            row_idx = torch.arange(qB, device=device).unsqueeze(1)
            new_index = merged_index[row_idx, new_pos]

            chunk_scores, chunk_index = new_scores, new_index

            # Clean up
            del t, scores, blk_scores, blk_idx, merged_scores, merged_index, new_scores, new_pos, new_index

        # Write chunk results back to CPU tensors
        top_scores_all[qs:qe] = chunk_scores.to("cpu")
        top_index_all[qs:qe] = chunk_index.to("cpu")

        del q, chunk_scores, chunk_index
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return top_index_all, top_scores_all


def build_result_dict(
    index_id: torch.Tensor,  # [N]
    top_indices: torch.Tensor,  # [N, k]
    top_scores: torch.Tensor,   # [N, k]
):
    """
    Build the desired Python dict structure from indices and scores.

    Output format:
        {
          index_id_i: {
             "candidate_ids": [...],
             "scores": [...],
             "ground_truth_id": index_id_i,
          },
          ...
        }
    """
    if index_id.ndim == 2 and index_id.size(1) == 1:
        index_id = index_id.squeeze(1)
    assert index_id.ndim == 1

    index_id = index_id.to("cpu")
    top_indices = top_indices.to("cpu")
    top_scores = top_scores.to("cpu")

    N, k = top_indices.shape
    result = {}

    for i in range(N):
        gt_id = int(index_id[i].item())
        cand_indices = top_indices[i].tolist()   # indices in [0, N)
        cand_scores = top_scores[i].tolist()     # floats

        # Map indices -> ids
        candidate_ids = [int(index_id[j].item()) for j in cand_indices]

        result[gt_id] = {
            "candidate_ids": candidate_ids,
            "scores": cand_scores,
            "ground_truth_id": gt_id,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Hard-negative mining: top-k cosine from safetensors.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to safetensors file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output pickle (.pkl).")
    parser.add_argument("--k", type=int, default=100, help="Top-k candidates to retrieve per query.")
    parser.add_argument("--q-chunk", type=int, default=4096, help="Query chunk size.")
    parser.add_argument("--t-chunk", type=int, default=65536, help="Target chunk size.")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (default: auto).")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 matmul on GPU.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    embeddings = [x for x in  os.listdir(args.input_dir) if x.endswith(".safetensors")]

    for embedding in embeddings:

        dataset_name = embedding.split(".")[0]
        
        input_path = os.path.join(args.input_dir, embedding)
        output_path = os.path.join(args.output_dir, dataset_name+".pkl")
        if os.path.exists(output_path):
            continue


        print(f"Loading safetensors from {input_path}...")
        tensors = load_safetensors(str(input_path))

        # Expected keys: index_id, query, target
        try:
            index_id = tensors["index_id"]
            query = tensors["query"]
            target = tensors["target"]
        except KeyError as e:
            raise KeyError(
                f"Missing expected key in safetensors: {e}. "
                f"Expected keys: 'index_id', 'query', 'target'"
            ) from e

        print("Computing top-k cosine similarities...")
        top_indices, top_scores = compute_topk_cosine(
            query,
            target,
            k=args.k,
            q_chunk=args.q_chunk,
            t_chunk=args.t_chunk,
            device=args.device,
            use_fp16=not args.no_fp16,
        )

        print("Building result dict...")
        result = build_result_dict(index_id, top_indices, top_scores)

        print(f"Saving pickle to {output_path}...")
        with open(output_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done.")


if __name__ == "__main__":
    main()
