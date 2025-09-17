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
    embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    embedding_model = embedding_model.to(training_args.device)
    embedding_model.eval()

    os.makedirs(encode_output_path, exist_ok=True)
    # --- Main Evaluation Loop ---
    for dataset_idx, (dataset_name, task_config) in enumerate(dataset_configs.items()):

        total_correct = 0
        full_eval_qry_dataset, full_eval_cand_dataset = AutoPairEvalDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, processor=processor, **task_config)
        for row in full_eval_qry_dataset:

            query = row['query_text'][0]

            has_rewrites = query.split("assistant\n")[-1].replace("<|im_end|>", "").strip() != ""
            if has_rewrites:
                query = query.split("Answer:")[-1].split("Summary:")[-1]
            else:
                query = query.split(": ")[-1].replace("<|im_end|>\n<|im_start|>assistant\n<|im_end|>", "")\
                            .replace("<|im_start|>system\nYou are a helpful assistant specialized in multimodal embedding.<|im_end|>\n<|im_start|>user\n", "").strip()
            
            query = replace_all(query, "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>", "<|image_pad|>") 
            gt = row['cand_text'][0]
            gt = gt.split(": ")[-1].replace("<|im_end|>\n<|im_start|>assistant\n<|im_end|>", "")\
                            .replace("<|im_start|>system\nYou are a helpful assistant specialized in multimodal embedding.<|im_end|>\n<|im_start|>user\n", "").strip()
            gt = replace_all(gt, "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>", "<|image_pad|>") 

            if dataset_name in {"MVBench"}:
                query_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]
                query_choice = None
                for choice in query_choices:
                    if f"({choice})" in query or f"{choice}." in query or query == choice:
                        query_choice = choice
                        break
                if not query_choice:
                    print(f"Choice not found in query: {query}")
                gt_choices = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]
                gt_choice = None
                for choice in gt_choices:
                    if choice in gt:
                        gt_choice = choice.strip("()")
                        break
                if not gt_choice:
                    print(f"Choice not found in gt: {gt}")

                if query_choice == gt_choice:
                    total_correct += 1
                else:
                    print(f"Mismatch: {row['query_text'][0]} | {row['cand_text'][0]}")
            elif dataset_name in {"Video-MME", "ActivityNetQA"}:
                if query.lower() in gt.lower():
                    total_correct += 1
                else:
                    print(f"Mismatch: {row['query_text'][0]} | {row['cand_text'][0]}")
        print(f"Dataset: {dataset_name}, Accuracy: {total_correct/len(full_eval_qry_dataset):.4f}")
                    

if __name__ == "__main__":
    main()

