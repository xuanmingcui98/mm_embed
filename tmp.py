import json
import sys
import yaml
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
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor

import logging

logging.getLogger("PIL").setLevel(logging.WARNING)

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
            videos_batch.append(None)
        elif q_mm is not None and mod == "video":
            images_batch.append(None)
            videos_batch.append(q_mm)
        else:
            images_batch.append(None)
            videos_batch.append(None)

    inputs = processor(
        text=prompts,
        images=images_batch if any(x is not None for x in images_batch) else None,
        videos=videos_batch if any(x is not None for x in videos_batch) else None,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    gen_ids = model.generate(
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
    parser.add_argument("--is_student", action="store_true", help="Whether the model is a student model.")

    return parser.parse_args()


def main():
    args = parse_args()

    if 'qwen2.5' in args.model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        model_cls = Qwen2_5_VLForConditionalGeneration
        processor = Qwen2_5_VLProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    elif 'qwen2' in args.model_name.lower():
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        model_cls = Qwen2VLForConditionalGeneration
        processor = Qwen2VLProcessor.from_pretrained(args.model_name, max_pixels=1280 * 28 * 28, trust_remote_code=True)

    if args.is_student:
        from src.data.prompts import query_user_prompts_cot_generation as query_prompt_template, target_user_prompts_cot_generation as target_prompt_template
    else:
        from generation_prompts_qwen import query_prompt_template, target_prompt_template

    processor.padding_side = "left"
    model = model_cls.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        # _attn_implementation="flash_attention_2",
    )

    if args.lora_ckpt:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_ckpt)
        model = model.merge_and_unload()
    # processor = Autoprocessor.from_pretrained(args.model_name, trust_remote_code=True)

    with open(args.dataset_config, "r") as f:
        dataset_configs = yaml.safe_load(f)

    for _, config in dataset_configs.items():
        dataset_name = config["dataset_name"]
        image_dir = config.get("image_dir", config.get("frame_root", None))

        print(f"==> Processing dataset: {dataset_name}")

        args.model_name = args.model_name.strip("/")
        ckpt_name=args.model_name.split("/")[-1] if not args.lora_ckpt else args.lora_ckpt.split("/")[-1]
        if ckpt_name.startswith("checkpoint-"):
            ckpt_name = args.model_name.split("/")[-2] 
        folder = os.path.join(args.output_folder, "hf_inference", ckpt_name + "max_res_only", dataset_name, "cot")
        os.makedirs(folder, exist_ok=True)
        encode_side = config.get("encode_side", "query")
        if encode_side == 'query':
            prompt_template = query_prompt_template[dataset_name]
        else:
            prompt_template = target_prompt_template[dataset_name]

        if dataset_name in IMAGE_TASKS | VISDOC_TASKS:
            mm_modality = "image"
        else:
            mm_modality = "video"

        query_text_field = target_text_field = mm_field = None
        if dataset_name in IMAGE_TASKS:
            dataset = load_dataset("MMEB-eval", dataset_name, split="test")

            query_text_field = "processed_query_text"
            target_text_field = "processed_target_text"

            mm_field = "qry_img_path"
            if encode_side == "query":
                key_fields = ['qry_text', 'qry_img_path']
            else:
                key_fields = ['tgt_text', 'qry_img_path']

            image_dir = "/home/xuanmingcui/datasets/MMEB-eval/eval_images/"


            def func(row):
                row['processed_query_text'] = extract_query(row['qry_text'], dataset_name)
                row['processed_target_text'] = [extract_target(x, dataset_name) for x in row['tgt_text']]
                return row

            dataset = dataset.map(func, load_from_cache_file=False)

        if encode_side == "target" and isinstance(dataset[0][key_fields[0]], list):
            added = set()
            paired_dataset = []
            for row in dataset:
                for i in range(len(dataset[0][key_fields[0]])):
                    key = tuple([row[key_fields[x][i]] for x in range(len(key_fields))])
                    if key not in added:
                        added.add(key)
                    to_add = {**row}
                    for k,v in to_add.items():
                        if isinstance(v, list):
                            to_add[k] = v[i]
                    paired_dataset.append(to_add)

            paired_dataset = sorted(list(paired_dataset))
            dataset = datasets.Dataset.from_list(paired_dataset)

        if args.n_partitions > 1:
            dataset = dataset.shard(num_shards=args.n_partitions, index=args.current_partition-1)

        # load existing descriptions
        if encode_side == "target":
            pkl_files = [x for x in os.listdir(folder) if x.startswith("target") and x.endswith(".pkl")]
        else:
            pkl_files =  [x for x in os.listdir(folder) if x.endswith(".pkl")]
        descriptions = {}
        if len(pkl_files) > 0:
            print(f"Found existing descriptions in {folder}, loading...")
            for f in pkl_files:
                descriptions.update(pickle.load(open(os.path.join(folder, f), "rb")))

        if encode_side == "target":
            intermediate_files = [x for x in os.listdir(folder) if x.startswith("target") and x.endswith(".jsonl")]
        else:
            intermediate_files = [x for x in os.listdir(folder) if x.endswith(".jsonl")]
        if len(intermediate_files) > 0:
            for f in intermediate_files:
                for line in open(os.path.join(folder, f), "r"):
                    line = json.loads(line)
                    descriptions[tuple(line['key'])] = line["response"]

        dataset_unprocessed_idx = []

        for idx, row in enumerate(dataset):

            key = tuple([row[x] for x in key_fields])
                
            if key not in descriptions:
                dataset_unprocessed_idx.append(idx)
        
        dataset = dataset.select(dataset_unprocessed_idx)
                
        print(dataset)

        intermediates = open(os.path.join(folder, f"{encode_side}_intermediates_{args.current_partition}-{args.n_partitions}_{str(os.environ.get('SLURM_JOB_ID'))}.jsonl"), "a") 
        
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), args.batch_size)):

                batch = dataset[i:i + args.batch_size]
                bs = len(next(iter(batch.values())))
                dummy_inputs = [''] * bs
                query_text_inputs, target_text_inputs = batch.get(query_text_field, dummy_inputs), batch.get(target_text_field, dummy_inputs)
                text_inputs = query_text_inputs if encode_side == "query" else target_text_inputs
                text_inputs = [prompt_template.format(query=text_input) for text_input in text_inputs]
                modalities = batch['modality']
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
                
        pickle.dump(descriptions, open(os.path.join(folder, f"{encode_side}_descriptions_{args.current_partition}-{args.n_partitions}.pkl"), "wb"))
        intermediates.close()

        if args.current_partition == args.n_partitions:
            print(f"Finished processing dataset {dataset_name}.")
            # merge all pickles and intermediate files
            all_descriptions = {}
            pkl_files = [x for x in os.listdir(folder) if x.startswith(encode_side) and x.endswith(".pkl")]
            for f in pkl_files:
                all_descriptions.update(pickle.load(open(os.path.join(folder, f), "rb")))
            pickle.dump(all_descriptions, open(os.path.join(folder, f"{encode_side}.pkl"), "wb"))

if __name__ == "__main__":
    main()
