# Adapted from Tevatron code
import logging
import os.path
import sys
import os

os.environ['TZ'] = "America/Los_Angeles"

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)

import torch
import wandb
import yaml
import tqdm  
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


def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    set_seed(training_args.seed)

    processor = load_processor(model_args, data_args)

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)

    prompts = {}
    setattr(training_args, 'model_backbone', "qwen2_vl")
    setattr(model_args, "model_backbone", "qwen2_vl")
    train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args, processor)

    system_prompt = "\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    for row in train_dataset:
        if row['global_dataset_name'] not in prompts:
            print(f"Dataset: {row['global_dataset_name']}")
            print(f"Query: {row['query_text']}")
            print(f"Target: {row['pos_text']}")
            row['query_text'] = row['query_text'].replace(system_prompt, "").strip()
            row['pos_text'] = row['pos_text'].replace(system_prompt, "").strip()
            prompts[row['global_dataset_name']] = (row['query_text'], row['pos_text'])

    filename = "training_prompts_unified"
    if data_args.apply_chat_template:
        filename += "_chat"
    if data_args.query_description_dir or data_args.target_description_dir:
        filename += "_desc"
    json.dump(prompts, open(f"{filename}.json", 'w'), indent=4)


if __name__ == "__main__":
    main()
