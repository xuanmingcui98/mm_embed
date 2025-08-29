import json
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import pickle
import os
import argparse
from vllm import LLM, SamplingParams

import logging

logging.getLogger("PIL").setLevel(logging.WARNING)

# os.environ["HF_HOME"] = "/opt/dlami/nvme/xuanmingcui/.cache/huggingface"

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch


def parse_args():
    parser = argparse.ArgumentParser(description="Generate descriptions using a language model.")

    parser.add_argument("--n_partitions", type=int, default=1, help="Number of partitions for the dataset. Starting from 1.")
    parser.add_argument("--current_partition", type=int, default=1, help="Current partition index.")

    return parser.parse_args()

def main():

    args = parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    with open("descriptions/video_rewrites_300k.jsonl", "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    descriptions = pickle.load(open("descriptions/video_caption_300k-t2v/target.pkl", "rb"))
    data = [row for row in data if row["id"] not in descriptions]

    # divide by partition
    data = data[(args.current_partition - 1) * (len(data) // args.n_partitions) : args.current_partition * (len(data) // args.n_partitions)]

    llm = LLM(
        model=model_name,
        trust_remote_code=True,  # Required to load Phi-3.5-vision
        max_model_len=1024,  # Otherwise, it may not fit in smaller GPUs
        tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs
        enforce_eager=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
 
    if "internvl" in model_name.lower():
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    sampling_params = SamplingParams(max_tokens=1024)

    intermediates = open(f"descriptions/video_rewrites_300k_processed-{args.current_partition}_{args.n_partitions}.jsonl", "a")
    


    for i in tqdm(range(0, len(data), 64)):
        batch = data[i : i + 64]

        formatted_inputs = []

        for row in batch:

            formatted_inputs.append(
                {"prompt": tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful agent specialized in summarization."}, {"role": "user", "content": f"Generate a short, succinct, and concise one-sentence summary for the below video caption. Only include the summary in your answer. Video caption:\n\n{row['rewrite']}\n\nSummary: "}], add_generation_prompt=True, tokenize=False)}
            )
                
        responses = llm.generate(formatted_inputs, sampling_params=sampling_params,)

        for row, response in zip(batch, responses):
            summary = response.outputs[0].text
            full = f"{row['rewrite']}\nSummary: {summary}"
            descriptions[row["id"]] = full

            intermediates.write(json.dumps({"key": (), "response": response.outputs[0].text}) + "\n")
            intermediates.flush()
        
    intermediates.close()
    pickle.dump(descriptions, open(os.path.join("descriptions", "video_caption_300k-t2v", f"target_descriptions_{args.current_partition}-{args.n_partitions}.pkl"), "wb"))

if __name__ == "__main__":
    main()
