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
    parser.add_argument("--dataset_names", nargs="+", default=None)
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
    setattr(training_args, 'model_backbone', "qwen2_vl")
    setattr(model_args, "model_backbone", "qwen2_vl")

    set_seed(training_args.seed)

    processor = load_processor(model_args, data_args)

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), 'w'), indent=2)

    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        max_model_len=30_000,  
        task="score",
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=torch.cuda.device_count()==1
    )

    symmetric_instruction = "Given a web search query, retrieve relevant passages that have highest semantic similarity to the query (from most similar to least similar)"
    query_instruction = "Given a web search query, retrieve relevant passages that best answer the query"

    query_template = """<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}
    <Query>:{query}\n"""

    document_template = """<Document>: {target}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"""


    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    for k, v in dataset_configs.items():
        
        dataset_name = v.get("subset_name", v.get("dataset_name"))

        if args.dataset_names and dataset_name not in args.dataset_names: 
            continue

        print("==> Start processing: ", dataset_name)

        output_path = os.path.join(args.output_dir, f"{dataset_name}.pkl")
        if os.path.exists(output_path):
            print("==> Existed. Skipping: ", dataset_name)
            continue

        if dataset_name in VISDOC_TASKS:
            instruction = query_instruction
        else:
            instruction = symmetric_instruction

        results = {}
        
        train_dataset = init_mixed_dataset({k:v}, model_args, data_args, training_args, processor)
        
        i = 0
        iterator = iter(train_dataset)
        try:
            current_sample = next(iterator)
        except StopIteration:
            current_sample = None

        with tqdm(total=train_dataset.num_rows) as pbar:
            while current_sample is not None:
                current_cluster = current_sample["cluster"]
                cluster = [current_sample]

                # collect all samples with the same cluster id
                while True:
                    try:
                        next_sample = next(iterator)
                    except StopIteration:
                        next_sample = None
                    if next_sample is None or next_sample["cluster"] != current_cluster:
                        current_sample = next_sample  # next starting point
                        break
                    cluster.append(next_sample)

                # progress update
                pbar.update(len(cluster))
                i += len(cluster)

                # if current_cluster == -1:
                #     continue

                cluster_dict = {k: [d[k] for d in cluster] for k in cluster[0]}
                queries = (
                    cluster_dict["query_ecr"]
                    if cluster_dict["query_ecr"][0]
                    else [extract_query(x, dataset_name) for x in cluster_dict["query_text"]]
                )
                targets = (
                    cluster_dict["pos_ecr"]
                    if cluster_dict["pos_ecr"][0]
                    else [extract_target(x, dataset_name) for x in cluster_dict["pos_text"]]
                )

                queries = [query_template.format(query=q, instruction=instruction) for q in queries]
                targets = [document_template.format(target=t) for t in targets]

                for idx, query in enumerate(queries):
                    scores = llm.score(query, targets, use_tqdm=False)
                    scores = [x.outputs.score for x in scores]
                    results[cluster_dict["index_id"][idx]] = {
                        "cluster": cluster_dict["cluster"][idx],
                        "scores": scores,
                        "candidate_ids": cluster_dict["index_id"],
                        "ground_truth_id": cluster_dict["index_id"][idx],
                    }

        pkl.dump(results, open(output_path, 'wb'))

if __name__ == "__main__":
    main()


