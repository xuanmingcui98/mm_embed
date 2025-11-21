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
from typing import Tuple, List, Optional
from datasets import Dataset, IterableDataset

os.environ['TZ'] = "America/New_York"

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, AutoConfig
from safetensors.torch import save_file

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.loader.mixed_dataset import AutoPairDataset
from src.eval_utils.metrics import RankingMetrics
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor, COLPALI
from src.utils import batch_to_device, print_rank, print_master
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)


def encode_embeddings(
    model: MMEBModel,
    loader: DataLoader,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    full_dataset: Dataset,
) -> tuple[np.ndarray, list]:
    """
    Encodes embeddings for a given dataset using the model, handling both standard and
    late-interaction models in a DDP-safe manner.
    """
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    local_query_reps = []
    local_target_reps = []
    local_index_ids = []

    model.eval()
    with torch.no_grad():
        for inputs in tqdm(loader, disable=local_rank > 0):
            query_inputs = batch_to_device(inputs[0], training_args.device)
            target_inputs = batch_to_device(inputs[1], training_args.device)
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                # Determine if encoding query or target based on available keys
                query_reps, _ = model.encode_input(query_inputs)
                target_reps, _ = model.encode_input(target_inputs)
                query_reps = query_reps.detach().cpu().float()
                target_reps = target_reps.detach().cpu().float()

                local_query_reps.append(query_reps)
                local_target_reps.append(target_reps)
                local_index_ids.append(torch.tensor(query_inputs['index_id']))
    
    local_query_reps = torch.cat(local_query_reps, dim = 0)
    local_target_reps = torch.cat(local_target_reps, dim = 0)
    local_index_ids = torch.cat(local_index_ids, dim = 0).int()
    if dist.is_initialized() and full_dataset.num_rows >= world_size:
        gathered_query_reps = [None] * world_size
        gathered_target_reps = [None] * world_size
        gathered_index_ids = [None] * world_size
        dist.all_gather_object(gathered_query_reps, local_query_reps)
        dist.all_gather_object(gathered_target_reps, local_target_reps)
        dist.all_gather_object(gathered_index_ids, local_index_ids)

        gathered_query_reps = torch.cat(gathered_query_reps, dim=0)
        gathered_target_reps = torch.cat(gathered_target_reps, dim=0)
        gathered_index_ids = torch.cat(gathered_index_ids, dim=0)

    else:
        gathered_query_reps = local_query_reps
        gathered_target_reps = local_target_reps
        gathered_index_ids = local_index_ids

    
    return gathered_query_reps, gathered_target_reps, gathered_index_ids

        

def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--dataset_config", type=str, default="configs/train/train_alltasks.yaml", help="Path to the dataset configuration file")
    parser.add_argument("--output_dir", type=str, default="train_embeddings", help="Directory to save evaluation results")
    parser.add_argument("--query_description_dir", type=str, default=None, help="Directory containing dataset descriptions")
    parser.add_argument("--target_description_dir", type=str, default=None, help="Directory containing dataset descriptions")

    return parser.parse_args()

def main():
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # DEBUG PRINTS for Distributed Setup
    print_master("Distributed init debug info:")
    print_master(f"RANK: {os.environ.get('RANK')}")
    print_master(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print_master(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print_master(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print_master(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
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

    eval_args.target_description_dir = eval_args.query_description_dir if eval_args.target_description_dir is None else eval_args.target_description_dir
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
    # setattr(data_args, "resize_min_pixels", 56 * 56)
    # setattr(data_args, "resize_max_pixels", 2359296)

    # --- Model Loading ---
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not getattr(model_args, "model_backbone", None):
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
    
    setattr(training_args, 'model_backbone', model_args.model_backbone)
    print_master(f'Model Backbone: {model_args.model_backbone}')
    # --- DDP-Safe Model Loading ---
    # Step 1: Only the master process (rank 0) downloads the model.
    if local_rank == 0:

        model, processor = MMEBModel.load(model_args, data_args, is_trainable=False)
        print_master(f"[rank=0] Loading the model from Huggingface: {model_args.model_name}...")
    # Step 2: All processes wait here. The non-master processes will pause
    # until the master process (rank 0) finishes downloading and exits this barrier.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # Step 3: Now that the model is cached, the non-master processes load it from the local cache.
    if local_rank != 0:
        print_rank(f"Loading the model from cache...")
        time.sleep(random.randint(2 * local_rank, 3 * local_rank))
        model, processor = MMEBModel.load(model_args, data_args, is_trainable=False)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    encode_output_path = os.path.join(model_args.checkpoint_path, eval_args.output_dir)
    os.makedirs(encode_output_path, exist_ok=True)
    # --- Main Evaluation Loop ---
    for dataset_idx, (dataset_name, task_config) in enumerate(dataset_configs.items()):
        # try:
        # 0. load dataset
        if dist.is_initialized():
            dist.barrier()
        print_master(f"--- Evaluating {dataset_name} ---")

        if os.path.exists(os.path.join(encode_output_path, f"{dataset_name}_score.json")):
            print_master(f"Skipping {dataset_name} as it has already been evaluated.")
            continue

        embed_path = os.path.join(encode_output_path, f"{dataset_name}.safetensors")

        if os.path.exists(embed_path):
            continue

        if data_args.data_basedir is not None:
            # Construct full paths for data files if --data_basedir is provided
            for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                if data_args.data_basedir and task_config.get(key):
                    task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

        dataset = init_mixed_dataset({dataset_name:task_config}, model_args, data_args, training_args, processor)

        print_master("Encoding queries...")
        collator = MultimodalDataCollator(processor, model_args, data_args, training_args)
        dataloader = DataLoader(dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=collator, num_workers=training_args.dataloader_num_workers)
        gathered_query_reps, gathered_target_reps, gathered_index_ids = encode_embeddings(model, dataloader, training_args, model_args, dataset)

        if local_rank == 0:
            save_file({
                "query": gathered_query_reps,
                "target": gathered_target_reps,
                "index_id": gathered_index_ids
            }, embed_path)
            print_master(f"Saved query embeddings to {embed_path}")
        if dist.is_initialized():
            dist.barrier()

if __name__ == "__main__":
    main()

