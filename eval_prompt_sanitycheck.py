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

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, AutoConfig
from datasets import Dataset, concatenate_datasets
from datasets.distributed import split_dataset_by_node

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from src.data.eval_dataset.base_eval_dataset import AutoPairDataset, generate_cand_dataset
from src.eval_utils.metrics import RankingMetrics
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor, COLPALI
from src.utils import batch_to_device, print_rank, print_master
import multiprocessing
from multiprocessing import Pool, cpu_count
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)


def pad_dataset_to_divisible(dataset, world_size):
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
    padded_dataset = concatenate_datasets([dataset, padding_data])
    return padded_dataset, padded_size

def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--dataset_config", type=str, default="configs/eval/image.yaml", help="Path to the dataset configuration file")

    return parser.parse_args()

def main():
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if dist.is_initialized():
        print_rank(f"dist.get_rank(): {dist.get_rank()}")
        print_rank(f"dist.get_world_size(): {dist.get_world_size()}")

    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)

    eval_args = parse_eval_args()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    args = {}
    for js in ["model_args.json", "data_args.json", "training_args.json"]:
        if not os.path.exists(os.path.join(eval_args.checkpoint_path, js)) and "checkpoint-" in eval_args.checkpoint_path:
            # we have to one level up to get the args
            arg_path = os.path.dirname(eval_args.checkpoint_path)
        else:
            arg_path = eval_args.checkpoint_path
        args.update(json.load(open(os.path.join(arg_path, js), 'r')))

    args = args | vars(eval_args)  
    model_args, data_args, training_args = parser.parse_dict(args)
    training_args.debug = "prompt"
        
    output_path = os.path.join(model_args.checkpoint_path, "eval")
    os.makedirs(output_path, exist_ok=True)

    # --- Model Loading ---
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not getattr(model_args, "model_backbone", None):
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_master(f'Model Backbone: {model_args.model_backbone}')
    # --- DDP-Safe Model Loading ---
    # Step 1: Only the master process (rank 0) downloads the model.
    # if local_rank == 0:

    #     model, processor = MMEBModel.load(model_args, data_args, is_trainable=False)
    #     print_master(f"[rank=0] Loading the model from Huggingface: {model_args.model_name}...")
    # # Step 2: All processes wait here. The non-master processes will pause
    # # until the master process (rank 0) finishes downloading and exits this barrier.
    # if torch.distributed.is_initialized():
    #     torch.distributed.barrier()
    # # Step 3: Now that the model is cached, the non-master processes load it from the local cache.
    # if local_rank != 0:
    #     print_rank(f"Loading the model from cache...")
    #     time.sleep(random.randint(2 * local_rank, 3 * local_rank))
    #     model, processor = MMEBModel.load(model_args, data_args, is_trainable=False)

    processor = load_processor(model_args, data_args=data_args)
    # model.eval()
    # model = model.to(training_args.device, dtype=torch.bfloat16)
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    prompts = {}

    # --- Main Evaluation Loop ---
    for dataset_idx, (dataset_name, task_config) in enumerate(dataset_configs.items()):
        # 0. load dataset
        if dist.is_initialized():
            dist.barrier()
        print_master(f"--- Evaluating {dataset_name} ---")

        encode_output_path = os.path.join(model_args.checkpoint_path, "eval")
        os.makedirs(encode_output_path, exist_ok=True)

        query_embed_path = os.path.join(encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(encode_output_path, f"{dataset_name}_tgt")
        dataset_info_path = os.path.join(encode_output_path, f"{dataset_name}_info.jsonl")

        do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_embed_path)

        if do_query or do_cand:
            if data_args.data_basedir is not None:
                # Construct full paths for data files if --data_basedir is provided
                for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                    if data_args.data_basedir and task_config.get(key):
                        task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

            full_eval_qry_dataset, full_eval_cand_dataset = AutoPairDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, processor=processor, **task_config)

            eval_qry_dataset, eval_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset
            # Pad datasets to be divisible by world_size before splitting
            if dist.is_initialized():
                padded_qry_dataset, _ = pad_dataset_to_divisible(full_eval_qry_dataset, world_size)
                padded_cand_dataset, _ = pad_dataset_to_divisible(full_eval_cand_dataset, world_size)
                eval_qry_dataset = split_dataset_by_node(padded_qry_dataset, rank=local_rank, world_size=world_size)
                eval_cand_dataset = split_dataset_by_node(padded_cand_dataset, rank=local_rank, world_size=world_size)
            else:
                padded_qry_dataset, padded_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset

        # --- 1. Compute Query Embeddings ---
        print_master("Encoding queries...")
        eval_qry_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "qry")
        eval_qry_loader = DataLoader(eval_qry_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=eval_qry_collator, num_workers=training_args.dataloader_num_workers)
        for inputs, dataset_info in eval_qry_loader:

            print("Query inputs:")
            print(inputs['texts'][0])
            prompts[dataset_name] = inputs['texts'][0],
            break


        # --- 2. Compute Candidate Embeddings ---
        print_master("Encoding candidates...")
        eval_cand_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "cand")
        eval_cand_loader = DataLoader(eval_cand_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=eval_cand_collator, num_workers=training_args.dataloader_num_workers)

        for inputs, dataset_info in eval_cand_loader:

            print("Target inputs:")
            print(inputs['texts'][0])
            prompts[dataset_name] += (inputs['texts'][0],)
            break
    
    output_name = f"prompts_{os.path.basename(data_args.dataset_config).replace('.yaml', '')}"
    if data_args.apply_chat_template:
        output_name += "_chat"
    
    json.dump(prompts, open(f"{output_name}.json", 'w'), indent=4)

if __name__ == "__main__":
    main()

