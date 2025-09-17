import datetime
import logging
import json
import random
import time
import argparse
import numpy as np
import os
import pickle
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

os.environ['TZ'] = "America/New_York"

from transformers import AutoModel
from tqdm import tqdm
from transformers import HfArgumentParser, AutoConfig
from src.model.processor import get_backbone_name, load_processor, COLPALI
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from src.data.loader.mixed_dataset import AutoPairEvalDataset
from src.eval_utils.metrics import RankingMetrics

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)

def replace_all(text, *args):
    for a in args:
        text = text.replace(a, "")
    return text.strip()

def encode_embeddings(
    model,
    dataset,
    encode_key: str,
) -> tuple[np.ndarray, list]:
    """
    Encodes embeddings for a given dataset using the model, handling both standard and
    late-interaction models in a DDP-safe manner.
    """

    embeds = []
    keys = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), 512)):
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                # Determine if encoding query or target based on available keys
                batch = dataset[i : i + 512]
                inputs = [x[0] for x in batch[encode_key]]

                has_rewrites = inputs[0].split("assistant\n")[-1].replace("<|im_end|>", "").strip() != ""
                if has_rewrites:
                    inputs = [x.split("Answer:")[-1].split("Summary:")[-1] for x in inputs]
                else:
                    inputs = [x.split(": ")[-1].replace("<|im_end|>\n<|im_start|>assistant\n<|im_end|>", "")\
                              .replace("<|im_start|>system\nYou are a helpful assistant specialized in multimodal embedding.<|im_end|>\n<|im_start|>user\n", "").strip() for x in inputs]
                
                inputs = [replace_all(x, "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>", "<|image_pad|>") for x in inputs]
                output = model.encode(inputs, task="text-matching")

                if encode_key == "cand_text":
                    rank_keys = [x['cand_name'] for x in batch['dataset_infos']]
                else:
                    rank_keys = batch['dataset_infos']
                
                keys.extend(rank_keys)

            embeds.append(output)

    embeds = np.concatenate(embeds, axis=0)

    return embeds, keys


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=512, help="Batch size for evaluation")
    parser.add_argument("--dataset_config", type=str, default="configs/eval/image.yaml", help="Path to the dataset configuration file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--query_description_dir", type=str, default=None, help="Directory containing dataset descriptions")
    parser.add_argument("--target_description_dir", type=str, default=None, help="Directory containing dataset descriptions")

    return parser.parse_args()

def main():

    eval_args = parse_eval_args()

    eval_args.target_description_dir = eval_args.target_description_dir if eval_args.target_description_dir is not None else eval_args.query_description_dir
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_dict(vars(eval_args))
        
    # --- Model Loading ---
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not getattr(model_args, "model_backbone", None):
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    
    processor = load_processor(model_args, data_args=data_args)

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    encode_output_path = eval_args.output_dir if eval_args.output_dir is not None else os.environ.get("OUTPUT_DIR", eval_args.query_description_dir)
    encode_output_path = os.path.join(encode_output_path, f"gen_eval")
    embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    embedding_model = embedding_model.to(training_args.device)
    embedding_model.eval()
    os.makedirs(encode_output_path, exist_ok=True)
    
    qa_datasets =  ['A-OKVQA', 'HatefulMemes', 'ImageNet-A', 'Country211', 'ChartQA', 'ScienceQA', 'Visual7W', 'MSCOCO_i2t', 
                    'DocVQA', 'ObjectNet', 'VOC2007', 'N24News', 'GQA', 'SUN397', 
                    'ImageNet-R', 'InfographicsVQA', 'TextVQA', 'Place365', 'ImageNet-1K', 'VizWiz', 'OK-VQA',
                    'Video-MME', 'NExTQA', 'EgoSchema', 'MVBench', 'ActivityNetQA', 'Kinetics-700', 'SmthSmthV2', 'UCF101', 'HMDB51', 'Breakfast']
    
    # --- Main Evaluation Loop ---
    for dataset_idx, (_, task_config) in enumerate(dataset_configs.items()):

        dataset_name = task_config["dataset_name"]

        if dataset_name not in qa_datasets:
            print(f"Skipping {dataset_name} as it is not in the predefined QA datasets.")
            continue

        encode_side = task_config.get("encode_side", "query") 

        if not os.path.exists(os.path.join(eval_args.query_description_dir, dataset_name, "cot", f"{encode_side}.pkl")):
            print(f"{encode_side} descriptions for {dataset_name} not found in {eval_args.query_description_dir}, skipping...")
            continue

        print(f"--- Evaluating {dataset_name} ---")

        query_embed_path = os.path.join(encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(encode_output_path, f"{dataset_name}_tgt")
        dataset_info_path = os.path.join(encode_output_path, f"{dataset_name}_info.jsonl")
        score_path = os.path.join(encode_output_path, f"{dataset_name}_score.json")
        pred_path = os.path.join(encode_output_path, f"{dataset_name}_pred.jsonl")

        if os.path.exists(score_path):
            print(f"Score file {score_path} already exists. Skipping evaluation for {dataset_name}.")
            continue

        do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_embed_path)

        if do_query or do_cand:
            if data_args.data_basedir is not None:
                # Construct full paths for data files if --data_basedir is provided
                for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                    if data_args.data_basedir and task_config.get(key):
                        task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

            full_eval_qry_dataset, full_eval_cand_dataset = AutoPairEvalDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, processor=processor, **task_config)

            eval_qry_dataset, eval_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset

        # --- 1. Compute Query Embeddings ---
        print("Encoding queries...")

        query_embeds, gt_infos = encode_embeddings(embedding_model, eval_qry_dataset, encode_key="query_text")

        print("Encoding candidates...")
        cand_embeds, all_cand_ids = encode_embeddings(embedding_model, eval_cand_dataset, encode_key="cand_text")
        cand_embed_dict = {cand_id: embed for cand_id, embed in zip(all_cand_ids, cand_embeds)}



        pred_dicts = []

        rank_against_all_candidates = task_config.get("eval_type", "global") == "global"
        if rank_against_all_candidates:
            cand_keys = list(cand_embed_dict.keys())
            cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])
            # Handle late-interaction scoring
            if query_embeds.ndim == 3: # Query: [N_q, L_q, H] | Candidate: [N_c, L_c, H]
                qry_embed = torch.from_numpy(query_embeds)
                cand_embeds = [torch.from_numpy(np.array(t)) for t in cand_embeds]
                scores = processor.score(qry_embed, cand_embeds, batch_size=64)  # use ColPali score function
                ranked_candids = torch.argsort(-scores, dim=1).cpu().numpy().tolist()
            else: # Dense
                cosine_scores = np.dot(query_embeds, cand_embeds.T)
                ranked_candids = np.argsort(-cosine_scores, axis=1)
            for qid, (ranked_candid, gt_info) in tqdm(enumerate(zip(ranked_candids, gt_infos)), desc=f"Calculating scores for {dataset_name}"):
                rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None
                assert rel_scores is None or len(rel_docids) == len(rel_scores)
                pred_dicts.append({
                    "prediction": [cand_keys[i] for i in ranked_candid],
                    "label": rel_docids,
                    "rel_scores": rel_scores,
                })
        else:

            query_embeds = torch.from_numpy(query_embeds).cuda()
            for qid, (qry_embed, gt_info) in enumerate(tqdm(zip(query_embeds, gt_infos), 
                                                            total=len(query_embeds), 
                                                            desc=f"Calculating scores for {dataset_name}")):
                cand_embeds = np.stack([cand_embed_dict[key] for key in gt_info["cand_names"]])
                if query_embeds.ndim == 3: # Query: [N_q, L_q, H] | Candidate: [N_c, L_c, H]
                    qry_embed = torch.from_numpy(np.array(qry_embed)).unsqueeze(0)
                    cand_embeds = [torch.from_numpy(np.array(t)) for t in cand_embeds]
                    scores = processor.score(qry_embed, cand_embeds, batch_size=1024)  # use ColPali score function
                    ranked_candids = torch.argsort(-scores, dim=1).cpu().numpy().tolist()[0]
                else:
                    # cosine_score = np.dot(qry_embed, cand_embeds.T)
                    # ranked_candids = np.argsort(-cosine_score)
                    cand_embeds = torch.from_numpy(cand_embeds).cuda()
                    scores = F.cosine_similarity(qry_embed.unsqueeze(0), cand_embeds)
                    ranked_candids = torch.argsort(-scores, dim=0).cpu().numpy().tolist()
                    
                rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None

                assert rel_scores is None or len(rel_docids) == len(rel_scores)
                pred_dicts.append({
                    "prediction": [gt_info["cand_names"][i] for i in ranked_candids],
                    "label": rel_docids,
                    "rel_scores": rel_scores,
                })

        metrics_to_report = task_config["metrics"] if task_config.get("metrics", None) is not None else ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
        metrics = RankingMetrics(metrics_to_report)
        score_dict = metrics.evaluate(pred_dicts)
        formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
        score_dict["num_pred"] = len(pred_dicts)
        score_dict["num_data"] = len(gt_infos)
        print(f"Score of {dataset_name}:")
        print(formatted)
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            json.dump(score_dict, f, indent=4)
        with open(pred_path, "w") as f:
            for pred in pred_dicts:
                f.write(json.dumps(pred) + '\n')

        # remove the encoded embeddings to save space
        if os.path.exists(query_embed_path):
            os.remove(query_embed_path)
        if os.path.exists(cand_embed_path):
            os.remove(cand_embed_path)


if __name__ == "__main__":
    main()

