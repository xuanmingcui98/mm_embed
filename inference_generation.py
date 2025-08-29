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
from generation_prompts_qwen import prompts, moment_retrieval_query_prompts
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
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--n_partitions", type=int, default=1, help="Number of partitions for the dataset. Starting from 1.")
    parser.add_argument("--current_partition", type=int, default=1, help="Current partition index.")
    parser.add_argument("--encode_target", type=str, default=None, help="Encoding query/target.")
    parser.add_argument("--output_folder", type=str, default="descriptions", help="Folder to save descriptions.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")

    return parser.parse_args()


def main():
    args = parse_args()

    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        max_model_len=30_000,  
        limit_mm_per_prompt={"image": 1, "video":1},  
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=torch.cuda.device_count()==1,

    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    sampling_params = SamplingParams(max_tokens=1024)

    dataset_names = args.dataset_name.split(",")

    for dataset_name in dataset_names:
        dataset_name = dataset_name.strip()

        folder = os.path.join(args.output_folder, dataset_name)
        if os.path.exists(folder):
            print(f"Folder {folder} already exists, skipping...")
            continue
        print(f"==> Processing dataset: {dataset_name}")

        if dataset_name not in ["MomentSeeker", "Charades-STA", "QVHighlight"] + list(IMAGE_TASKS):
            prompt_template = prompts[dataset_name]
            prompt_template = prompt_template.replace("<image>", "<|image_pad|>").replace("<video>", "<|video_pad|>")
        


        if dataset_name in TRAIN_TASKS:
            with open("configs/train/train_alltasks.yaml", "r") as f:
                config = yaml.safe_load(f)
                if dataset_name in VIDEO_TASKS:
                    image_dir = config[dataset_name]["video_frame_basedir"]
                elif dataset_name in VISDOC_TASKS:
                    image_dir = None
                else:
                    image_dir = config[dataset_name]["image_dir"]
        else:
            if dataset_name in VISDOC_TASKS:
                with open("configs/eval/visdoc.yaml", "r") as f:
                    config = yaml.safe_load(f)
                    image_dir = config[dataset_name]["image_root"]
            elif dataset_name in VIDEO_TASKS:
                with open("configs/eval/video.yaml", "r") as f:
                    config = yaml.safe_load(f)
                    image_dir = config[dataset_name]["frame_root"]
            else:
                with open("configs/eval/image.yaml", "r") as f:
                    config = yaml.safe_load(f)
                    image_dir = config[dataset_name]["image_root"]

        if dataset_name in IMAGE_TASKS | VISDOC_TASKS:
            mm_modality = "image"
        else:
            mm_modality = "video"

        query_text_field = target_text_field = mm_field = None
        if dataset_name in IMAGE_TASKS:
            dataset = load_dataset("MMEB-eval", dataset_name, split="test")
            query_text_field = "qry_text"
            target_text_field = "tgt_text"
            mm_field = "qry_img_path"

            # if dataset_name in ["VisualNews_i2t", "MSCOCO_i2t"]:
            #     args.encode_target = "query"
            # elif dataset_name in ["VisualNews_t2i", "MSCOCO_t2i"]:
            #     args.encode_target = 'target'
            #     key_fields = ['tgt_img_path']
            # elif dataset_name in ["ImageNet-1K", "ImageNet_1K", "ImageNet-A", "ImageNet-R", "VOC2007", "ObjectNet"]:
            #     args.encode_target = "query"

            args.encode_target = "query"
            key_fields = ['qry_text', 'qry_img_path']
            image_dir = "/home/xuanmingcui/datasets/MMEB-eval/eval_images/"

        elif dataset_name == "openbmb/VisRAG-Ret-Train-In-domain-data":
            dataset = load_dataset(dataset_name, split="train")
            key_fields = ['query', 'image', 'source']
            mm_field = "image"
            args.encode_target = "target"
        elif dataset_name == "vidore/colpali_train_set":
            dataset = load_dataset(dataset_name, split="train")
            key_fields = ['query', 'image_filename', 'answer']
            query_text_field = "query"
            target_text_field = "answer"
            mm_field = 'image'
            args.encode_target = "query"
        elif dataset_name == "ActivityNetQA":
            dataset = load_dataset('json', data_files=config["data_path"])['train']
            key_fields = ['question', 'video_name']
            query_text_field = "question"
            mm_field = 'video_name'
            args.encode_target = "query"
            def func(row):
                row['question'] = format_qa_with_choices(row['question'], ['yes', 'no'])
                return row
            dataset = dataset.map(func)
        elif dataset_name == "DiDeMo":
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
            key_fields = ['video_rel']
            def func(row):
                row['video_filename'] = os.path.splitext(os.path.basename(row['video']))[0]
                return row
            dataset = dataset.map(func)
            args.encode_target = "target"
        elif dataset_name == "EgoSchema":
            dataset = load_dataset("lmms-lab/egoschema", "Subset", split="test")
            key_fields = ['question', 'video_idx']
            mm_field = 'video_idx'
            query_text_field = "question"
            args.encode_target = "query"
            def func(row):
                row['question'] = format_qa_with_choices(row['question'], row['option'])
                return row
        elif dataset_name in ["QVHighlight", "Charades-STA"]:
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
            def func(row):
                row['video_filename'] = os.path.splitext(os.path.basename(row['video_path']))[0]
                if args.encode_target == 'query':
                    row['video_filename'] = os.path.join(row['video_filename'], "query")
                else:
                    row['video_filename'] = [os.path.join(row['video_filename'], filename) for filename in os.listdir(os.path.join(image_dir, row['video_filename'])) if filename != "query"]
                return row
            dataset = dataset.map(func)
            if args.encode_target == 'query':
                key_fields = ["query", "video_filename"]
                query_text_field = "query"
                mm_field = 'video_filename'
                prompt_template = moment_retrieval_query_prompts[dataset_name]
            else:
                key_fields = ["video_filename"]
                mm_field = 'video_filename'
                query_text_field = None
                target_text_field = None
                prompt_template = prompts['video_caption_300k']
        elif dataset_name == "MomentSeeker":
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
            if args.encode_target == "query":
                key_fields = ['query', 'input_frames']
                mm_field = "input_frames"
                query_text_field = "query"
                prompt_template = moment_retrieval_query_prompts[dataset_name]
                def func(row):
                    input_frames = row['input_frames']
                    if isinstance(input_frames, str) and input_frames.endswith(".mp4"):
                        row['input_frames'] = os.path.join("video_frames", input_frames.split(".mp4")[0].replace("/", "_"))
                        row['mm_modality'] = "video"
                    elif isinstance(input_frames, str) and input_frames.endswith(".jpg"):
                        row['input_frames'] = "query_" + row['input_frames']
                        row['mm_modality'] = "image"
                    else:
                        row = None # nothing to rewrite 

                    return row
            else:
                key_fields = ["video_filename"]
                prompt_template = prompts['video_caption_300k']
                def func(row):
                    raw_frames = row["positive_frames"] + row["negative_frames"]
                    row['video_filename'] = []
                    for file in raw_frames:
                        clip_name = file['output_path'].replace("/", "_").split(".mp4")[0]
                        clip_name = os.path.join("video_frames", clip_name)
                        row['video_filename'].append(clip_name)
                    row['mm_modality'] = "video"
                    return row
            dataset = dataset.map(func)

        elif dataset_name in ['MSR-VTT', 'MSVD']:
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
            mm_field = "video_id"
            key_fields = ['video_id']
            args.encode_target = "target"
        elif dataset_name == "MVBench":
            args.encode_target = "query"

            from src.data.eval_dataset.mvbench_dataset import subset_meta
            def func(row):
                subset         = row["subset"]
                query_raw      = row["question"]
                video_filename = row["video"]
                cands_raw      = row["candidates"]
                answer_raw     = row["answer"]

                _, cand_text, answer, answer_idx = qa_template(query_raw, cands_raw, answer_raw)
                query_text = format_qa_with_choices(query_raw, cand_text)
                row['query'] = query_text
                row['video_filename'] = os.path.join(subset, video_filename)
                return row
            
            subsets = []
            for subset_name in subset_meta.keys():
                dataset = load_dataset("OpenGVLab/MVBench", subset_name, split="train")
                new_column = [subset_name] * len(dataset)
                dataset = dataset.add_column("subset", new_column)
                subsets.append(dataset)
            dataset = datasets.concatenate_datasets(subsets)

            dataset = dataset.map(func)

            mm_field = "video_filename"
            query_text_field = "query"
            key_fields = ['question', 'subset', 'video']
        
        elif dataset_name == 'NExTQA':
            args.encode_target = "query"
            dataset = load_dataset("lmms-lab/NExTQA", "MC", split="test")
            mm_field = "video"
            key_fields = ['question', 'video']

            def func(row):
                video_id = row["video"]
                query    = row["question"]
                answer   = row["answer"]      # index
                qid      = row["qid"]
                _type    = row["type"]
                a0       = row["a0"]
                a1       = row["a1"]
                a2       = row["a2"]
                a3       = row["a3"]
                a4       = row["a4"]

                options = [a0, a1, a2, a3, a4]
                _, cand_text, _, answer_idx = qa_template(query, options, answer)
                query_text = format_qa_with_choices(query, cand_text)
                row['query'] = query_text
                return row
        
            dataset = dataset.map(func)
        
        elif dataset_name == "SmthSmthV2":
            args.encode_target = "query"

            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
            dataset = sample_dataset(dataset, num_sample_per_subset=1000)
            mm_field = "video_id"
            key_fields = ['video_id']
        elif dataset_name == "VATEX":
            args.encode_target = "target"
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
            mm_field = "videoID"
            key_fields = ['videoID']
        elif dataset_name in ["HMDB51", "UCF101", "Kinetics-700", "Breakfast"]:
            args.encode_target = "query"
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
            mm_field = "video_id"
            key_fields = ['video_id']
        elif dataset_name == "Video-MME":
            args.encode_target = "query"
            dataset = load_dataset("lmms-lab/Video-MME", split="test")
            mm_field = "videoID"
            key_fields = ['question', 'videoID']
            query_text_field = "query_text"

            def func(row):
                row['query_text'] = row['question'] + "\n" + "\n".join(row['options'])
                return row
            dataset = dataset.map(func)

        elif dataset_name in VIDORE_QA_RETRIEVAL_DATASETS:
            args.encode_target = "target"
            hf_dataset_name = EVAL_DATASET_HF_PATH[dataset_name][0]
            hf_dataset_split = EVAL_DATASET_HF_PATH[dataset_name][2]
            qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
            corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
            dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
            lang = EVAL_DATASET_HF_PATH[dataset_name][1]
            if lang is not None:
                dataset = dataset.filter(lambda example: example["language"] == lang)
            qrels_mapping = load_qrels_mapping(qrels)
            dataset_dict = {"image": [], "corpus-id": []}

            for row in tqdm(dataset, desc=f"Processing {dataset_name} dataset"):
                query_id = row['query-id']
                for corpus_id, _ in qrels_mapping[query_id].items():
                    dataset_dict["image"].append(str(corpus_id) + ".png")
                    dataset_dict["corpus-id"].append(corpus_id)
            
            for row in corpus:
                if row['corpus-id'] not in dataset_dict["corpus-id"]:
                    dataset_dict["corpus-id"].append(row['corpus-id'])
                    dataset_dict["image"].append(str(row['corpus-id']) + ".png")
                
            dataset = datasets.Dataset.from_dict(dataset_dict)
            
            key_fields = ['corpus-id']
            mm_field = 'image'
        
        elif dataset_name in VISRAG_QA_RETRIEVAL_DATASETS:
            args.encode_target = "target"
            hf_dataset_name = EVAL_DATASET_HF_PATH[dataset_name][0]
            hf_dataset_split = EVAL_DATASET_HF_PATH[dataset_name][2]
            qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
            corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
            dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
            qrels_mapping = load_qrels_mapping(qrels)
            lang = EVAL_DATASET_HF_PATH[dataset_name][1]
            if lang is not None:
                dataset = dataset.filter(lambda example: example["language"] == lang)

            dataset_dict = {"image": [], "image_filename": []}
            for row in tqdm(dataset, desc=f"Processing {dataset_name} dataset"):
                query_id = row['query-id']
                for image_name, rel_score in qrels_mapping[query_id].items():
                    base, ext = os.path.splitext(image_name)
                    short_base = base[:50] + "_" + hashlib.md5(image_name.encode("utf-8")).hexdigest()[:8]
                    new_imagename = short_base + ext
                    dataset_dict["image"].append(new_imagename)
                    dataset_dict["image_filename"].append(new_imagename)

            for row in corpus:
                if row['corpus-id'] not in dataset_dict["image_filename"]:
                    image_name = row['corpus-id']
                    base, ext = os.path.splitext(image_name)
                    short_base = base[:50] + "_" + hashlib.md5(image_name.encode("utf-8")).hexdigest()[:8]
                    new_imagename = short_base + ext
                    dataset_dict["image_filename"].append(new_imagename)
                    dataset_dict["image"].append(new_imagename)

            dataset = datasets.Dataset.from_dict(dataset_dict)

            key_fields = ['image_filename']
            mm_field = 'image'

        elif dataset_name == "YouCook2":
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
            mm_field = 'id'
            key_fields = ['id']

        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

        # momentseeker has mixed modality
        if dataset_name not in ['MomentSeeker']:
            dataset = dataset.add_column("modality", [mm_modality] * len(dataset))

        if args.encode_target == "target" and isinstance(dataset[0][key_fields[0]], list):
            paired_dataset = set()
            for row in dataset:
                for i in range(len(dataset[0][key_fields[0]])):
                    paired_dataset.add(tuple([row[key_fields[x][i]] for x in range(len(key_fields))]))
            
            paired_dataset = sorted(list(paired_dataset))
            dataset = datasets.Dataset.from_dict({
                key: [x[i] for x in paired_dataset] for i, key in enumerate(key_fields)
            })

        if args.n_partitions > 1:
            dataset = dataset.shard(num_shards=args.n_partitions, index=args.current_partition-1)

        folder = os.path.join(args.output_folder, dataset_name, "cot")
        os.makedirs(folder, exist_ok=True)

        # load existing descriptions
        if args.encode_target == "target":
            pkl_files = [x for x in os.listdir(folder) if x.startswith("target") and x.endswith(".pkl")]
        else:
            pkl_files =  [x for x in os.listdir(folder) if x.endswith(".pkl")]
        descriptions = {}
        if len(pkl_files) > 0:
            print(f"Found existing descriptions in {folder}, loading...")
            for f in pkl_files:
                descriptions.update(pickle.load(open(os.path.join(folder, f), "rb")))

        if args.encode_target == "target":
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

        intermediates = open(os.path.join(folder, f"{args.encode_target}_intermediates_{args.current_partition}-{args.n_partitions}_{str(os.environ.get('SLURM_JOB_ID'))}.jsonl"), "a") 
        
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), args.batch_size)):

                batch = dataset[i:i + args.batch_size]
                bs = len(next(iter(batch.values())))
                dummy_inputs = [''] * bs
                query_text_inputs, target_text_inputs = batch.get(query_text_field, dummy_inputs), batch.get(target_text_field, dummy_inputs)
                if dataset_name in IMAGE_TASKS:
                    text_inputs = [x.replace("<|image_1|>\n", "") for x in query_text_inputs]
                else:
                    text_inputs = [prompt_template.format(query=query_text, target=target_text) for query_text, target_text in zip(query_text_inputs, target_text_inputs)]
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
                        # read as a numpy array
                        # np_frames = [cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB) for frame_path in frame_paths]
                        mm_inputs.append([Image.open(frame_path) for frame_path in frame_paths])
                        # mm_inputs.append(np.array(np_frames))
                    elif isinstance(mm_input, Image.Image):
                        mm_inputs.append(mm_input)
                    else:
                        raise ValueError(f"Unsupported mm_inputs type: {type(mm_inputs[0])}")

                formatted_inputs = []

                for qry_text, qry_mm, modality in zip(text_inputs, mm_inputs, modalities):
                    if qry_mm is not None:
                        video_messages = [
                            # {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [
                                    {"type": "text", "text": qry_text},
                                    {"type": modality, modality: qry_mm} 
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
                
        pickle.dump(descriptions, open(os.path.join(folder, f"{args.encode_target}_descriptions_{args.current_partition}-{args.n_partitions}.pkl"), "wb"))
        intermediates.close()

        if args.current_partition == args.n_partitions:
            print(f"Finished processing dataset {dataset_name}.")
            # merge all pickles and intermediate files
            all_descriptions = {}
            pkl_files = [x for x in os.listdir(folder) if x.startswith(args.encode_target) and x.endswith(".pkl")]
            for f in pkl_files:
                all_descriptions.update(pickle.load(open(os.path.join(folder, f), "rb")))
            pickle.dump(all_descriptions, open(os.path.join(folder, f"{args.encode_target}.pkl"), "wb"))

if __name__ == "__main__":
    main()
