import json
import sys
import yaml
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor
import torch
from tqdm import tqdm
import pickle
import os
from datasets import load_dataset
import re
from typing import List, Optional, Sequence, Union
import datasets
import PIL
from PIL import Image
import hashlib
import argparse

from src.data.prompts import (IMAGE_TASKS, 
                              VIDEO_TASKS, 
                              VISDOC_TASKS, 
                              VIDORE_QA_RETRIEVAL_DATASETS, 
                              VISRAG_QA_RETRIEVAL_DATASETS, 
                              TRAIN_TASKS)
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset, load_qrels_mapping
from src.data.utils.vision_utils import process_video_frames, qa_template
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.prompts import extract_query, extract_target
import cv2
import numpy as np
from accelerate import Accelerator
from contextlib import nullcontext
from src.data.generation_utils import prepare_generation_dataset, get_unprocessed_data

import logging

logging.getLogger("PIL").setLevel(logging.WARNING)

# os.environ["HF_HOME"] = "/opt/dlami/nvme/xuanmingcui/.cache/huggingface"

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

def format_qa_with_choices(query, choices):
    return query + "\nOptions:\n" + "\n".join(choices)

def build_messages(q_text: str,
                   q_mm: Optional[Union["PIL.Image.Image", Sequence["PIL.Image.Image"], "np.ndarray"]],
                   modality: Optional[str]):
    """
    Builds a Qwen2-VL chat message. `modality` should be "image" or "video" (or None for text-only).
    For video, pass either:
      - a list of PIL Images (sampled frames), or
      - a 4D numpy/tensor shaped like (T, H, W, C) in RGB.
    """
    if q_mm is not None and modality in {"image", "video"}:
        content = [
            {"type": "text", "text": q_text},
            {"type": modality, modality: q_mm},
        ]
    else:
        content = [{"type": "text", "text": q_text}]
    return [{"role": "user", "content": content}]

@torch.inference_mode()
def hf_mm_generate_qwen2vl(
    model,
    accelerator,
    processor,
    text_inputs: List[str],
    mm_inputs: List[Optional[object]],            # images or videos aligned with text_inputs
    modalities: List[Optional[str]],              # "image" | "video" | None for each sample
    max_new_tokens: int = 512,
    
):
    """
    Batched multimodal generation using Hugging Face .generate().
    Handles mixed batches of text-only, image, and video queries.
    """
    prompts: List[str] = []
    images_batch: List[Optional[object]] = []
    videos_batch: List[Optional[object]] = []

    for q_text, q_mm, mod in zip(text_inputs, mm_inputs, modalities):
        msgs = build_messages(q_text, q_mm, mod)
        prompt = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)

        if q_mm is not None and mod == "image":
            images_batch.append(q_mm)
        elif q_mm is not None and mod == "video":
            videos_batch.append(q_mm)

    inputs = processor(
        text=prompts,
        images=images_batch if any(x is not None for x in images_batch) else None,
        videos=videos_batch if any(x is not None for x in videos_batch) else None,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    gen_ids = accelerator.unwrap_model(model).generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    gen_ids = gen_ids[:, inputs["input_ids"].shape[1]:]  
    texts = processor.batch_decode(gen_ids, skip_special_tokens=True)

    return texts

def parse_args():
    parser = argparse.ArgumentParser(description="Generate descriptions using a language model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--lora_ckpt", type=str, default=None, help="Name of the model to use.")
    parser.add_argument("--dataset_config", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--n_partitions", type=int, default=1, help="Number of partitions for the dataset. Starting from 1.")
    parser.add_argument("--current_partition", type=int, default=1, help="Current partition index.")
    parser.add_argument("--output_folder", type=str, default="descriptions", help="Folder to save descriptions.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--use_gt", action="store_true", help="Whether to include GT in the prompt.")
    parser.add_argument("--use_cot", action="store_true", help="Whether to use chain-of-thought prompting.")
    parser.add_argument("--is_student", action="store_true", help="Whether the model is a student model.")

    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if 'qwen2.5' in args.model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        model_cls = Qwen2_5_VLForConditionalGeneration
        processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    elif 'qwen2' in args.model_name.lower():
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        model_cls = Qwen2VLForConditionalGeneration
        # processor = Qwen2VLProcessor.from_pretrained(args.model_name, trust_remote_code=True)
        # processor = Qwen2VLProcessor.from_pretrained(args.model_name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28, trust_remote_code=True)
        # processor = Qwen2VLProcessor.from_pretrained(args.model_name, max_pixels=1280 * 28 * 28, trust_remote_code=True)
        processor = Qwen2VLProcessor.from_pretrained(args.model_name, max_pixels=2359296, trust_remote_code=True)

    if args.is_student:
        print0("Using student model prompts. Forcing use_cot to be True and use_gt to be False.")
        args.use_cot = True
        args.use_gt = False

    if args.use_gt:
        from generation_prompts_qwen_with_gt import query_prompt_template, target_prompt_template
    elif args.is_student:
        from src.data.prompts import query_user_prompts_cot_generation as query_prompt_template, target_user_prompts_cot_generation as target_prompt_template
    else:
        if args.use_cot:
            from generation_prompts_qwen import query_prompt_template, target_prompt_template
        else:
            from generation_prompts_qwen import query_prompt_no_cot as query_prompt_template, target_prompt_template

    processor.padding_side = "left"
    model = model_cls.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        # _attn_implementation="flash_attention_2",
    )

    if args.lora_ckpt:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_ckpt)
        model = model.merge_and_unload()

    model = accelerator.prepare(model)
    device = accelerator.device
    model.eval()

    world_size = accelerator.num_processes
    rank = accelerator.process_index

    total_shards = max(1, args.n_partitions) * world_size
    shard_index = (max(0, args.current_partition - 1) * world_size) + rank

    with open(args.dataset_config, "r") as f:
        dataset_configs = yaml.safe_load(f)

    for _, config in dataset_configs.items():
        dataset_name = config["dataset_name"]
        encode_side = config.get("encode_side", "query")
        
        args.model_name = args.model_name.strip("/")
        ckpt_name=args.model_name.split("/")[-1] if not args.lora_ckpt else args.lora_ckpt.split("/")[-1]
        if ckpt_name.startswith("checkpoint-"):
            ckpt_name = args.model_name.split("/")[-2] 
        folder = os.path.join(args.output_folder, f"hf_inference_ddp{"_use_gt" if args.use_gt else ''}{"_use_cot" if args.use_cot else ''}{"_isstudent" if args.is_student else ''}", ckpt_name + "max_res_only", dataset_name, "cot")
        os.makedirs(folder, exist_ok=True)

        if os.path.exists(os.path.join(folder, f"{encode_side}.pkl")):
            print0(f"Descriptions for dataset {dataset_name} already exist, skipping...")
            continue

        print0(f"==> Processing dataset: {dataset_name}")

        if encode_side == 'query':
            prompt_template = query_prompt_template[dataset_name]
        else:
            prompt_template = target_prompt_template[dataset_name]

        if dataset_name in IMAGE_TASKS | VISDOC_TASKS:
            mm_modality = "image"
        else:
            mm_modality = "video"

        dataset_info = prepare_generation_dataset(config)

        dataset = dataset_info["dataset"]
        mm_field = dataset_info["mm_field"]
        query_text_field = dataset_info["query_text_field"]
        target_text_field = dataset_info["target_text_field"]
        key_fields = dataset_info["key_fields"]
        image_dir = dataset_info['image_dir']

        dataset, descriptions = get_unprocessed_data(dataset, encode_side, key_fields, folder)
        
        # debug
        # dataset = dataset.select(range(1))

        if total_shards > 1:
            dataset = dataset.shard(num_shards=total_shards, index=shard_index)
                
        print0(dataset)

        intermediates = open(os.path.join(folder, f"{encode_side}_intermediates_{args.current_partition}-{args.n_partitions}_rank{rank}_ws{world_size}_{str(os.environ.get('SLURM_JOB_ID'))}.jsonl"), "a") 
        
        amp_ctx = accelerator.autocast() if hasattr(accelerator, "autocast") else nullcontext()
        with torch.no_grad(), amp_ctx:
            rng = range(0, len(dataset), args.batch_size)
            pbar = tqdm(rng, disable=not accelerator.is_local_main_process)
            for i in pbar:

                batch = dataset[i:i + args.batch_size]
                modalities = batch['modality']
                if dataset_name == 'MomentSeeker' and args.is_student:
                    prompt_templates = [prompt_template[mod] for mod in modalities]
                else:
                    prompt_templates = [prompt_template] * len(modalities)
                bs = len(next(iter(batch.values())))
                dummy_inputs = [''] * bs
                query_text_inputs, target_text_inputs = batch.get(query_text_field, dummy_inputs), batch.get(target_text_field, dummy_inputs)
                text_inputs = query_text_inputs if encode_side == "query" else target_text_inputs
                text_inputs = [pt.format(query=text_input) for text_input, pt in zip(text_inputs, prompt_templates)]
                raw_mm_inputs = batch.get(mm_field)
                mm_inputs = []

                for idx, mm_input in enumerate(raw_mm_inputs):
                    
                    if not mm_input:
                        mm_inputs.append(None)
                        continue

                    mm_modality = batch['modality'][idx]
                    mm_input = os.path.join(image_dir, mm_input) if isinstance(mm_input, str) else mm_input
                    if isinstance(mm_input, str) and mm_modality == "image":
                        mm_inputs.append(Image.open(os.path.join(image_dir, mm_input)))
                    elif isinstance(mm_input, str) and mm_modality == "video" and os.path.isdir(mm_input):
                        frame_paths = process_video_frames(mm_input, num_frames=8)
                        mm_inputs.append([Image.open(frame_path) for frame_path in frame_paths])
                    elif isinstance(mm_input, Image.Image):
                        mm_inputs.append(mm_input)
                    else:
                        raise ValueError(f"Unsupported mm_inputs type: {type(mm_inputs[0])}")

                responses = hf_mm_generate_qwen2vl(
                    model,
                    accelerator,
                    processor,
                    text_inputs,
                    mm_inputs,
                    modalities,
                )

                keys = [tuple([batch[x][i] for x in key_fields]) for i in range(bs)]

                for key, response in zip(keys, responses):
                    descriptions[key] = response

                    intermediates.write(json.dumps({"key": key, "response": response}) + "\n")
                    intermediates.flush()
                
        # pickle.dump(descriptions, open(os.path.join(folder, f"{encode_side}_descriptions_{args.current_partition}-{args.n_partitions}.pkl"), "wb"))
        intermediates.close()
        accelerator.wait_for_everyone()

        if args.current_partition == args.n_partitions and is_main:
            print0(f"Finished processing dataset {dataset_name}.")
            # merge all pickles and intermediate files
            all_descriptions = {}
            pkl_files = [x for x in os.listdir(folder) if x.startswith(encode_side) and x.endswith(".pkl")]
            jsonl_files = [x for x in os.listdir(folder) if x.startswith(f"{encode_side}_intermediates_") and x.endswith(".jsonl")]
            for f in pkl_files:
                all_descriptions.update(pickle.load(open(os.path.join(folder, f), "rb")))

            for f in jsonl_files:
                for line in open(os.path.join(folder, f), "r"):
                    line = json.loads(line)
                    all_descriptions[tuple(line['key'])] = line["response"]
            pickle.dump(all_descriptions, open(os.path.join(folder, f"{encode_side}.pkl"), "wb"))

if __name__ == "__main__":
    main()
