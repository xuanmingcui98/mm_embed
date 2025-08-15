import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor, AutoConfig, AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.eval_utils import get_pred
from src.utils import print_rank
from src.model_utils import get_backbone_name
# from internvl_inference import split_model, load_image
from qwen_inference import batch_inference as batch_inference_qwen
import re
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
import datasets
from lmdeploy.vl.constants import IMAGE_TOKEN
from datasets import load_from_disk
from src.model import MMEBModel
from src.model_utils import load_processor, get_backbone_name, process_vlm_inputs_fns
from src.dataset import TrainTextImageDataset, EvalDataset
from src.collator import TrainTextImageDataCollator, EvalCollator
import functools
import torch
from torch.utils.data import DataLoader, Subset
from copy import deepcopy

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch


@torch.no_grad()
def main():
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
    os.makedirs(data_args.encode_output_path, exist_ok=True)
    model = MMEBModel.build(model_args, training_args)
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args)
    setattr(model, 'processor', processor)

    print(f'model_args: {json.dumps(vars(model_args), indent=2)}')
    print(f'data_args: {json.dumps(vars(data_args), indent=2)}')

    model = model.to(training_args.device)
    model.eval()
    all_subsets = data_args.subset_name

    process_fn = functools.partial(process_vlm_inputs_fns[model_backbone], processor=processor, max_length=data_args.max_len)

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(data_args.subset_name):

        latents_train = {}
        latents_test = {}

        print(f"\033[91m{idx+1}/{len(all_subsets)}: Processing {subset} now!\033[0m")

        data_args.subset_name = [subset]

        data_args.dataset_name = "TIGER-Lab/MMEB-train"
        data_args.split_name = ["original"]
        data_args.dataset_split = "train"
        data_args.image_dir = "/home/xuanmingcui/datasets/MMEB-train"
        train_dataset = TrainTextImageDataset(data_args, model_args)
        train_collator = TrainTextImageDataCollator(data_args, model_args, processor)

        # slice based on current_partition and n_partitions
        train_indices = list(range(len(train_dataset) * (data_args.current_partition - 1) // data_args.n_partitions,
                                     len(train_dataset) * data_args.current_partition // data_args.n_partitions))
        train_dataset = Subset(train_dataset, train_indices)
        print_rank(f"Subset {subset} train dataset size: {len(train_dataset)}")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=train_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        data_args = deepcopy(data_args)

        data_args.dataset_name = "TIGER-Lab/MMEB-eval"
        data_args.split_name = ["test"]
        data_args.dataset_split = "test"
        data_args.image_dir = "/home/xuanmingcui/datasets/MMEB-eval/eval_images/"

        
        eval_qry_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
        )
        eval_tgt_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
        )

        eval_qry_indices = list(range(len(eval_qry_dataset) * (data_args.current_partition - 1) // data_args.n_partitions,
                                        len(eval_qry_dataset) * data_args.current_partition // data_args.n_partitions))
        eval_qry_dataset = Subset(eval_qry_dataset, eval_qry_indices)
        print_rank(f"Subset {subset} eval qry dataset size: {len(eval_qry_dataset)}")

        eval_tgt_indices = list(range(len(eval_tgt_dataset) * (data_args.current_partition - 1) // data_args.n_partitions,
                                        len(eval_tgt_dataset) * data_args.current_partition // data_args.n_partitions))
        eval_tgt_dataset = Subset(eval_tgt_dataset, eval_tgt_indices)
        print_rank(f"Subset {subset} eval tgt dataset size: {len(eval_tgt_dataset)}")

        eval_collator = EvalCollator(
            data_args=data_args,
            model_args=model_args,
            processor=processor,
        )

        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            num_workers=training_args.dataloader_num_workers,
        )

        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Processing {subset} train data")):
            
            qry_input, tgt_input = batch
            qry_input = batch_to_device(process_fn(qry_input), training_args.device)
            tgt_input = batch_to_device(process_fn(tgt_input), training_args.device)

            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                qry_out = model.encoder(**qry_input, return_dict=True, output_hidden_states=True).hidden_states
            qry_hidden_states = torch.stack(qry_out, dim=0).permute(1, 0, 2, 3)  # (batch_size, num_layers, seq_len, hidden_size)
            qry_hidden_states = qry_hidden_states[:, -data_args.last_n_hidden_states:, :, :]  # (batch_size, last_n_layers, seq_len, hidden_size)
            if model_args.normalize:
                qry_hidden_states = torch.nn.functional.normalize(qry_hidden_states, dim=-1)
            for text, img_path, latent, attention_mask in zip(batch[0]['text'], batch[0]['image_path'], qry_hidden_states, qry_input['attention_mask']):
                # only take the latents with non-zero attention mask
                latent = latent[:, attention_mask.bool(), :]
                latents_train[(text, img_path)] = latent.cpu().detach().float()

            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                tgt_out = model.encoder(**tgt_input, return_dict=True, output_hidden_states=True).hidden_states
            tgt_hidden_states = torch.stack(tgt_out, dim=0).permute(1, 0, 2, 3)  # (batch_size, num_layers, seq_len, hidden_size)
            tgt_hidden_states = tgt_hidden_states[:, -data_args.last_n_hidden_states:, :, :]  # (batch_size, last_n_layers, seq_len, hidden_size)
            if model_args.normalize:
                tgt_hidden_states = torch.nn.functional.normalize(tgt_hidden_states, dim=-1)
            for text, img_path, latent, attention_mask in zip(batch[1]['text'], batch[1]['image_path'], tgt_hidden_states, tgt_input['attention_mask']):
                latent = latent[:, attention_mask.bool(), :]
                latents_train[(text, img_path)] = latent.cpu().detach().float()


        torch.cuda.empty_cache()

        for i, batch in enumerate(tqdm(eval_qry_loader, desc=f"Processing {subset} eval qry data")):

            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                qry_out = model.encoder(**batch_to_device(batch, training_args.device), return_dict=True, output_hidden_states=True).hidden_states
            qry_hidden_states = torch.stack(qry_out, dim=0).permute(1, 0, 2, 3)  # (batch_size, num_layers, seq_len, hidden_size)
            qry_hidden_states = qry_hidden_states[:, -data_args.last_n_hidden_states:, :, :]  # (batch_size, last_n_layers, seq_len, hidden_size)
            if model_args.normalize:
                qry_hidden_states = torch.nn.functional.normalize(qry_hidden_states, dim=-1)
            for text, img_path, latent, attention_mask in zip(batch['texts'], batch['img_path'], qry_hidden_states, batch['attention_mask']):
                latent = latent[:, attention_mask.bool(), :]
                latents_test[(text, img_path)] = latent.cpu().detach().float()

        torch.cuda.empty_cache()

        for i, batch in enumerate(tqdm(eval_tgt_loader, desc=f"Processing {subset} eval tgt data")):
            tgt_input = batch_to_device(batch, training_args.device)
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                tgt_out = model.encoder(**tgt_input, return_dict=True, output_hidden_states=True).hidden_states
            tgt_hidden_states = torch.stack(tgt_out, dim=0).permute(1, 0, 2, 3)  # (batch_size, num_layers, seq_len, hidden_size)
            tgt_hidden_states = tgt_hidden_states[:, -data_args.last_n_hidden_states:, :, :]  # (batch_size, last_n_layers, seq_len, hidden_size)
            if model_args.normalize:
                tgt_hidden_states = torch.nn.functional.normalize(tgt_hidden_states, dim=-1)
            for text, img_path, latent, attention_mask in zip(batch['texts'], batch['img_path'], tgt_hidden_states, batch['attention_mask']):
                latent = latent[:, attention_mask.bool(), :]
                latents_test[(text, img_path)] = latent.cpu().detach().float()

        # Save the latents
        encode_train_path = os.path.join(data_args.encode_output_path, subset, model_args.model_name.split("/")[-1], f"train_{data_args.current_partition}-{data_args.n_partitions}.pt")
        encode_test_path = os.path.join(data_args.encode_output_path, subset, model_args.model_name.split("/")[-1], f"test_{data_args.current_partition}-{data_args.n_partitions}.pt")
        os.makedirs(os.path.dirname(encode_train_path), exist_ok=True)

        torch.save(latents_train, encode_train_path)
        torch.save(latents_test, encode_test_path)

if __name__ == "__main__":
    main()
