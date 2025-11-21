import json
import sys
import yaml
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoTokenizer
import torch
from tqdm import tqdm
import pickle
import os
from datasets import load_dataset
import re
import datasets
from src.eval_utils.metrics import RankingMetrics
from vllm import LLM, SamplingParams
from PIL import Image
import hashlib
import argparse
from src.data.prompts import (IMAGE_TASKS, 
                              VIDEO_TASKS, 
                              VISDOC_TASKS, 
                              VIDORE_QA_RETRIEVAL_DATASETS, 
                              VISRAG_QA_RETRIEVAL_DATASETS, 
                              TRAIN_TASKS)
import numpy as np
from llm_output_parser import parse_json
import logging
import re, ast

def fuzzy_extract_list(s: str):
    # find the first [...]-like substring (optionally with whitespace)
    match = re.search(r'\[[^\[\]]*\]', s)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group())
    except Exception:
        return None

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

def format_qa_with_choices(query, choices):
    return query + "\nChoose your answer from the following options:\n" + " ".join(choices)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate descriptions using a language model.")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model to use.")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--dataset_names", type=str, default=None, nargs="+")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation.")
    parser.add_argument("--top_k", type=int, default=10, help="Batch size for generation.")
    parser.add_argument("--rerank_dir", type=str, default=None)

    # parser.add_argument("--n_partitions", type=int, default=1, help="Number of partitions for the dataset. Starting from 1.")
    # parser.add_argument("--current_partition", type=int, default=1, help="Current partition index.")

    return parser.parse_args()

symmetric_instruction = "Given a web search query, retrieve relevant passages that have highest semantic similarity to the query (from most similar to least similar)"
query_instruction = "Given a web search query, retrieve relevant passages that best answer the query"

query_template = """<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}
<Query>:{query}\n"""

document_template = """<Document>: {target}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"""

def main():
    args = parse_args()

    print(args)

    # llm = LLM(
    #     model=args.model_name,
    #     trust_remote_code=True,
    #     max_model_len=30_000,  
    #     task="score",
    #     hf_overrides={
    #         "architectures": ["Qwen3ForSequenceClassification"],
    #         "classifier_from_token": ["no", "yes"],
    #         "is_original_qwen3_reranker": True,
    #     },
    #     tensor_parallel_size=torch.cuda.device_count(),
    #     enforce_eager=torch.cuda.device_count()==1
    # )

    pred_files = [x for x in os.listdir(args.result_dir) if x.endswith("pred.jsonl")]
    if args.dataset_names:
        pred_files = [x for x in pred_files if x.replace("_pred.jsonl", "") in args.dataset_names]


    output_dir = args.rerank_dir +  "_results"
    os.makedirs(output_dir, exist_ok=True)
    pattern = r'"order"\s*:\s*\[\s*([^\]]*?)\s*\]'

    for filename in pred_files:

        dataset_name = filename.replace("_pred.jsonl", "")

        if os.path.exists(os.path.join(output_dir, filename)):
            print(f"already run {dataset_name}, skipping ...")
            continue

        reranking_dict = None
        if args.rerank_dir:
            if not os.path.exists(os.path.join(args.rerank_dir, f"{dataset_name}.pkl")):
                continue
            reranking_dict = pickle.load(open(os.path.join(args.rerank_dir, f"{dataset_name}.pkl"), "rb"))

        # if os.path.exists(os.path.join(output_dir, filename)):
        #     print(f"Already evaluated {dataset_name}, skipping ...")
        #     continue

        print(f"==> Processing dataset: {dataset_name}")

        dataset = load_dataset("json", data_files=os.path.join(args.result_dir, filename))['train']
                
        print(dataset)

        reranked_predictions = []

        for row in dataset:
            query_id = row['query_id'][0]
            reranked_result = reranking_dict.get((query_id,))
            if not reranked_result:
                reranked_predictions.append(row['prediction'])
            else:
                try:
                    rerank_result = parse_json(reranked_result)
                except:
                    rerank_result = re.search(pattern, reranked_result, flags=re.DOTALL).group(1)
                    rerank_result = [int(x) for x in re.findall(r'-?\d+', rerank_result)]
                if isinstance(rerank_result, dict):
                    rerank_result = rerank_result['order']
                reranked_prediction = [row['prediction'][i] for i in rerank_result]
                reranked_predictions.append(reranked_prediction)


        if "original_prediction" in dataset.features:
            dataset = dataset.rename_column("prediction", "reranked_prediction")
            dataset = dataset.add_column("prediction", reranked_predictions)
        else:
            dataset = dataset.rename_column("prediction", "original_prediction")
            dataset = dataset.add_column("prediction", reranked_predictions)

        pred_dicts = []

        for row in dataset:
            pred_dicts.append(row)

        metrics_to_report = ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
        metrics = RankingMetrics(metrics_to_report)
        score_dict = metrics.evaluate(pred_dicts)
        formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
        score_dict["num_pred"] = len(pred_dicts)
        print(f"Score of {dataset_name}:")
        print(formatted)
        with open(os.path.join(output_dir, f"{dataset_name}_score.json"), "w") as f:
            json.dump(score_dict, f, indent=4)
if __name__ == "__main__":
    main()