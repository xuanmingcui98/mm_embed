import json
import sys
import yaml
import pickle as pkl
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
    
    dataset = load_dataset("friedrichor/MSVD")['test']
    reranking_dict = pkl.load(open("gemini_full_rerank/MSVD.pkl", 'rb'))
    pattern = r'"order"\s*:\s*\[\s*([^\]]*?)\s*\]'

    print(dataset)

    reranked_predictions = []

    for row in dataset:
        reranked_result = reranking_dict.get((row['video_id'],))
        if not reranked_result:
            reranked_predictions.append(row['prediction'])
        else:
            try:
                rerank_result = parse_json(reranked_result)
            except:
                rerank_result = re.search(pattern, reranked_result, flags=re.DOTALL)
                if not rerank_result:
                    rerank_result = list(range(10))
                else:
                    rerank_result = rerank_result.group(1)
                    rerank_result = [int(x) for x in re.findall(r'-?\d+', rerank_result)]
            if isinstance(rerank_result, dict):
                rerank_result = rerank_result.get("order", [1])
            reranked_prediction = [dataset[i]['video_id'] for i in rerank_result]
            reranked_predictions.append(reranked_prediction)


    pred_dicts = []

    for pred, row in zip(reranked_predictions, dataset):
        pred_dicts.append({"prediction": pred, "label": [row['video_id']]} )

    metrics_to_report = ["hit", "precision", "recall", "f1", "map", "mrr"]
    metrics = RankingMetrics(metrics_to_report)
    score_dict = metrics.evaluate(pred_dicts)
    formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
    score_dict["num_pred"] = len(pred_dicts)
    print(formatted)
    # with open(os.path.join(output_dir, f"{dataset_name}_score.json"), "w") as f:
    #     json.dump(score_dict, f, indent=4)
if __name__ == "__main__":
    main()