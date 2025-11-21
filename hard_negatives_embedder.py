# Adapted from Tevatron code
import logging
import os.path
import sys
import os
import numpy as np
import pickle as pkl

os.environ['TZ'] = "America/New_York"

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)

import torch
import wandb
import yaml
from tqdm import tqdm  
import json
from functools import wraps
from transformers import HfArgumentParser, set_seed
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.model import MMEBModel
from src.trainer import GradCacheLateProcessTrainer, MMEBTrainer
from src.utils import print_rank, print_master, find_latest_checkpoint
from src.model.processor import get_backbone_name, load_processor, COLPALI
from src.data.prompts import extract_query, extract_target
import argparse
from vllm import LLM
from src.data.prompts import (IMAGE_TASKS, 
                              VIDEO_TASKS, 
                              VISDOC_TASKS, 
                              VIDORE_QA_RETRIEVAL_DATASETS, 
                              VISRAG_QA_RETRIEVAL_DATASETS, 
                              TRAIN_TASKS)

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']


    
original_tqdm_init = tqdm.__init__

@wraps(original_tqdm_init)
def new_tqdm_init(self, *args, **kwargs):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() > 0:
        kwargs['disable'] = True
        
    return original_tqdm_init(self, *args, **kwargs)

tqdm.__init__ = new_tqdm_init

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-8B", help="Name of the model to use.")
    parser.add_argument("--cluster_path", type=str, default="/fsx/yuhao/clusters_rewrite")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--use_query_rewrite", action="store_true")
    parser.add_argument("--description_dir", type=str, default="/fsx/xuanmingcui/descriptions_synced/")
    parser.add_argument("--dataset_config", type=str, default="configs/train/train_tmp.yaml")
    parser.add_argument("--output_dir", type=str, default="reranker_train")
    # parser.add_argument("--cluster_size", type=int, default=32)

    return parser.parse_args()

def main():
    args = parse_args()

    model_args, data_args, training_args = ModelArguments(), DataArguments(), TrainingArguments()

    data_args.cluster_path = args.cluster_path
    data_args.query_description_dir = args.description_dir
    data_args.target_description_dir = args.description_dir
    data_args.apply_chat_template = False
    data_args.dataset_config = args.dataset_config
    data_args.debug_prompt = True
    setattr(training_args, 'model_backbone', "qwen2_vl")
    setattr(model_args, "model_backbone", "qwen2_vl")

    set_seed(training_args.seed)

    processor = load_processor(model_args, data_args)

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), 'w'), indent=2)

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    for k, v in dataset_configs.items():

        dataset_name = v.get("subset_name", "dataset_name")

        results = {}
        
        train_dataset = init_mixed_dataset({k:v}, model_args, data_args, training_args, processor, is_iterable=False)
        
        current_cluster = train_dataset[0]['cluster']
        i = 0
        with tqdm(total=len(train_dataset), desc="Processing clusters") as pbar:
            while i < len(train_dataset):
                current_cluster = train_dataset[i]['cluster']
                end = i + 1

                # Move until the cluster changes or dataset ends
                while end < len(train_dataset) and train_dataset[end]['cluster'] == current_cluster:
                    end += 1

                cluster = train_dataset[i:end]

                # Update progress bar and index
                pbar.update(end - i)
                i = end

                # if current_cluster == -1: continue

                queries = cluster['query_ecr'] if cluster['query_ecr'][0] else [extract_query(x, dataset_name) for x in cluster['query_text']]
                targets = cluster['pos_ecr'] if cluster['pos_ecr'][0] else [extract_target(x, dataset_name) for x in cluster['pos_text']]

                for idx, query in enumerate(queries):
                    scores = [x.outputs.score for x in scores]
                    results[cluster['index_id'][idx]] = {
                        "cluster": cluster['cluster'][idx],
                        "scores": scores,
                        "candidate_ids": cluster['index_id'],
                        "ground_truth_id": cluster['index_id'][idx],
                    }

        pkl.dump(results, open(os.path.join(args.output_dir, f"{dataset_name}.pkl")))


if __name__ == "__main__":
    main()


