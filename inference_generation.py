import json
import sys
import yaml
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoTokenizer
import torch
from tqdm import tqdm
import pickle
import os
from datasets import load_dataset
import re
import datasets
from vllm import LLM, SamplingParams
from PIL import Image
import hashlib
import argparse
from src.data.prompts import extract_query, extract_target
from src.data.prompts import (IMAGE_TASKS, 
                              VIDEO_TASKS, 
                              VISDOC_TASKS, 
                              VIDORE_QA_RETRIEVAL_DATASETS, 
                              VISRAG_QA_RETRIEVAL_DATASETS, 
                              TRAIN_TASKS)
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset, load_qrels_mapping
from src.data.utils.vision_utils import process_video_frames, qa_template
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
import cv2
import numpy as np
from qwen_vl_utils import process_vision_info
from src.data.generation_utils import prepare_generation_dataset, get_unprocessed_data
import logging

logging.getLogger("PIL").setLevel(logging.WARNING)

# os.environ["HF_HOME"] = "/opt/dlami/nvme/xuanmingcui/.cache/huggingface"

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
    return query + "\nChoose your answer from the following options:\n" + " ".join(choices)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate descriptions using a language model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--n_partitions", type=int, default=1, help="Number of partitions for the dataset. Starting from 1.")
    parser.add_argument("--current_partition", type=int, default=1, help="Current partition index.")
    parser.add_argument("--output_folder", type=str, default="descriptions", help="Folder to save descriptions.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--dataset_config", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--use_gt", action="store_true", help="Whether to include GT in the prompt.")
    parser.add_argument("--use_cot", action="store_true", help="Whether to use chain-of-thought prompting.")
    parser.add_argument("--is_student", action="store_true", help="Whether the model is a student model.")

    return parser.parse_args()


def main():
    args = parse_args()

    print(args)

    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        max_model_len=30_000,  
        limit_mm_per_prompt={"image": 1, "video":1},  
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=torch.cuda.device_count()==1
    )
    sampling_params = SamplingParams(max_tokens=1024)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    assert not (args.is_student and args.use_gt), "Cannot use GT for student model."
    assert (not args.is_student) or args.use_cot, "Student model must use CoT."

    if args.use_gt:
        from generation_prompts_qwen_with_gt import query_prompt_template, target_prompt_template
    elif args.is_student:
        from src.data.prompts import query_user_prompts_cot_generation as query_prompt_template, target_user_prompts_cot_generation as target_prompt_template
    else:
        if args.use_cot:
            from generation_prompts_qwen import query_prompt_template, target_prompt_template
        else:
            from generation_prompts_qwen import query_prompt_no_cot as query_prompt_template, target_prompt_template


    with open(args.dataset_config, "r") as f:
        dataset_configs = yaml.safe_load(f)

    for _, config in dataset_configs.items():
        dataset_name = config["dataset_name"]
        image_dir = config.get("image_dir", config.get("frame_root", None))

        print(f"==> Processing dataset: {dataset_name}")

        folder = os.path.join(args.output_folder, "vllm", args.model_name, dataset_name, "cot")
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

        dataset_info = prepare_generation_dataset(config)

        dataset = dataset_info["dataset"]
        mm_field = dataset_info["mm_field"]
        query_text_field = dataset_info["query_text_field"]
        target_text_field = dataset_info["target_text_field"]
        key_fields = dataset_info["key_fields"]
        image_dir = dataset_info['image_dir']

        dataset, descriptions = get_unprocessed_data(dataset, encode_side, key_fields, folder)

        if args.n_partitions > 1:
            dataset = dataset.shard(num_shards=args.n_partitions, index=args.current_partition-1)

        # debug
        # dataset = dataset.select(range(1))
                
        print(dataset)

        print(prompt_template)

        intermediates = open(os.path.join(folder, f"{encode_side}_intermediates_{args.current_partition}-{args.n_partitions}_{str(os.environ.get('SLURM_JOB_ID'))}.jsonl"), "a") 
        
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), args.batch_size)):

                batch = dataset[i:i + args.batch_size]
                bs = len(next(iter(batch.values())))
                modalities = batch['modality']
                if dataset == 'MomentSeeker':
                    prompt_templates = [prompt_template[mod] for mod in modalities]
                else:
                    prompt_templates = [prompt_template] * len(modalities)
                dummy_inputs = [''] * bs
                query_text_inputs, target_text_inputs = batch.get(query_text_field, dummy_inputs), batch.get(target_text_field, dummy_inputs)
                text_inputs = query_text_inputs if encode_side == "query" else target_text_inputs
                if args.use_gt:
                    target_texts = [x[0] if isinstance(x, list) else x for x in target_text_inputs]
                    text_inputs = [pt.format(query=text_input, target=target) for text_input, target, pt in zip(text_inputs, target_texts, prompt_templates)]
                else:
                    text_inputs = [pt.format(query=text_input) for text_input, pt in zip(text_inputs, prompt_templates)]
                
                raw_mm_inputs = batch.get(mm_field)
                mm_inputs = []
                if raw_mm_inputs is not None:
                    for idx, mm_input in enumerate(raw_mm_inputs):
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

                formatted_inputs = []

                for qry_text, qry_mm, modality in zip(text_inputs, mm_inputs, modalities):
                    if qry_mm is not None:
                        video_messages = [
                            {"role": "user", "content": [
                                    {"type": "text", "text": qry_text},
                                    {"type": modality, modality: qry_mm,} 
                                ]
                            },
                        ]
                        image_inputs, video_inputs = process_vision_info(video_messages)

                        qry_mm = image_inputs if modality == "image" else video_inputs
                        formatted_inputs.append(
                            {"prompt": tokenizer.apply_chat_template(video_messages, add_generation_prompt=True, tokenize=False),
                            "multi_modal_data": {mm_modality: qry_mm}}
                        )
                    else:
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [
                                    {"type": "text", "text": qry_text},
                                ]
                            },
                        ]
                        formatted_inputs.append(
                            {"prompt": tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)}
                        )
                
                responses = llm.generate(formatted_inputs, sampling_params=sampling_params,)

                keys = [tuple([batch[x][i] for x in key_fields]) for i in range(bs)]

                for key, response in zip(keys, responses):
                    descriptions[key] = response.outputs[0].text

                    intermediates.write(json.dumps({"key": key, "response": response.outputs[0].text}) + "\n")
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
