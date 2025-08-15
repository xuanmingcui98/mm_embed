# Adapted from Tevatron code
import logging
import sys
import torch
import wandb

from transformers import (
    HfArgumentParser,
)
import json
from src.dataset import TrainTextImageDataset
from src.collator import TrainTextImageDataCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.model_utils import load_processor, get_backbone_name
from src.trainer import GradCacheLateProcessTrainer, MMEBTrainer
from src.utils import print_rank, print_master
from torch.utils.data import DataLoader
from src.model_utils import process_vlm_inputs_fns
import functools
from tqdm import tqdm


logger = logging.getLogger(__name__)


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

    model = MMEBModel.build(model_args, training_args)
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args)
    setattr(model, 'processor', processor)
    process_fn = functools.partial(process_vlm_inputs_fns[model_backbone], processor=processor, max_length=data_args.max_len)
    # model_args.model_backbone = "qwen2_vl"

    train_dataset = TrainTextImageDataset(data_args, model_args)
    collator = TrainTextImageDataCollator(data_args, model_args, processor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        num_workers=training_args.dataloader_num_workers)
    
    max_len = 0

    max_pixel_values = 0
    
    for inputs in tqdm(train_dataloader):
        for batch in inputs:
            inputs = process_fn(batch)

            pixel_values = inputs.get('pixel_values', None)
            if pixel_values is not None:
                if sum(pixel_values.shape) > max_pixel_values:
                    max_pixel_values = sum(pixel_values.shape)
                    max_pixel_values_shape = pixel_values.shape
                    max_images = inputs['images']

            input_ids = inputs['input_ids']
            if input_ids is not None:
                max_len = max(max_len, input_ids.shape[1])
    
    print(f'max_len: {max_len}')
    print(f'max_pixel_values_shape: {max_pixel_values_shape}')
    print(max_images)


if __name__ == "__main__":
    main()