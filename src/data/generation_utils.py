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

def format_qa_with_choices(query, choices):
    return query + "\nOptions:\n" + "\n".join(choices)

def prepare_generation_dataset(config):
    dataset_name = config.get("subset_name", config["dataset_name"])
    query_text_field = target_text_field = mm_field = None
    encode_side = config.get("encode_side", "query")
    image_dir = config.get("image_dir") or config.get("frame_root") or config.get("image_root")

    if dataset_name in IMAGE_TASKS | VISDOC_TASKS:
        mm_modality = "image"
    else:
        mm_modality = "video"

    if dataset_name in IMAGE_TASKS:
        dataset = load_dataset("MMEB-eval", dataset_name, split="test")

        query_text_field = "processed_query_text"
        target_text_field = "processed_target_text"

        mm_field = "qry_img_path" if encode_side == "query" else "tgt_img_path"
        if encode_side == "query":
            key_fields = ['qry_text', 'qry_img_path']
        else:
            key_fields = ['tgt_text', 'tgt_img_path']

        image_dir = "/home/xuanmingcui/datasets/MMEB-eval/eval_images/"

        # dataset = load_dataset("MMEB-train", dataset_name, split="original")
        # query_text_field = "processed_query_text"
        # target_text_field = "pos_text"
        # mm_field = "qry_image_path"   
        # encode_side = "query"
        # key_fields = ['qry', 'qry_image_path']
        # image_dir = "/home/xuanmingcui/datasets/MMEB-train"

        def func(row):
            row['processed_query_text'] = extract_query(row['qry_text'], dataset_name)
            row['processed_target_text'] = [extract_target(x, dataset_name) for x in row['tgt_text']]
            return row

        dataset = dataset.map(func, load_from_cache_file=False)

    elif dataset_name == "openbmb/VisRAG-Ret-Train-In-domain-data":
        dataset = load_dataset(dataset_name, split="train")
        key_fields = ['query', 'image', 'source']
        mm_field = "image"

    elif dataset_name == "vidore/colpali_train_set":
        dataset = load_dataset(dataset_name, split="train")
        key_fields = ['query', 'image_filename', 'answer']
        query_text_field = "query"
        target_text_field = "answer"
        mm_field = 'image'

    elif dataset_name == "ActivityNetQA":
        dataset = load_dataset('json', data_files=config["data_path"])['train']
        key_fields = ['question', 'video_name']
        query_text_field = "query_text"
        target_text_field = "answer"
        mm_field = 'query_mm_path'

        def func(row):
            row['query_text'] = format_qa_with_choices(row['question'], ['yes', 'no'])
            row['query_mm_path'] = os.path.join(config['frame_root'], f"v_{row['video_name']}")
            return row
        dataset = dataset.map(func, load_from_cache_file=False)
    elif dataset_name == "DiDeMo":
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        key_fields = ['video']
        query_text_field = "caption"
        def func(row):
            row['query_mm_path'] = os.path.splitext(os.path.basename(row['video']))[0]
            return row
        dataset = dataset.map(func, load_from_cache_file=False)
        mm_field = "query_mm_path"

    elif dataset_name == "EgoSchema":
        dataset = load_dataset("lmms-lab/egoschema", "Subset", split="test")
        key_fields = ['question', 'video_idx']
        mm_field = 'video_idx'
        query_text_field = "query_text"
        target_text_field = "pos_text"

        def func(row):
            row['query_text'] = format_qa_with_choices(row['question'], row['option'])
            row['pos_text'] = row['option'][int(row['answer'])]
            return row

        dataset = dataset.map(func, load_from_cache_file=False)
    elif dataset_name in ["QVHighlight", "Charades-STA"]:
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        def func(row):
            row['video_filename'] = os.path.splitext(os.path.basename(row['video_path']))[0]
            if encode_side == 'query':
                row['video_filename'] = os.path.join(row['video_filename'], "query")
            else:
                row['video_filename'] = [os.path.join(row['video_filename'], filename) for filename in os.listdir(os.path.join(image_dir, row['video_filename'])) if filename != "query"]
            return row
        dataset = dataset.map(func, load_from_cache_file=False)
        if encode_side == 'query':
            key_fields = ["query", "video_filename"]
            query_text_field = "query"
            mm_field = 'video_filename'

        else:
            key_fields = ["video_filename"]
            mm_field = 'video_filename'
            query_text_field = None
            target_text_field = None

    elif dataset_name == "MomentSeeker":
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        if encode_side == "query":
            key_fields = ['query', 'input_frames']
            mm_field = "input_frames"
            query_text_field = "query"

            def func(row):
                input_frames = row['input_frames']
                if isinstance(input_frames, str) and input_frames.endswith(".mp4"):
                    row['input_frames'] = os.path.join("video_frames", input_frames.split(".mp4")[0].replace("/", "_"))
                    row['modality'] = "video"
                elif isinstance(input_frames, str) and input_frames.endswith(".jpg"):
                    row['input_frames'] = "query_" + row['input_frames']
                    row['modality'] = "image"
                else:
                    row['input_frames'] = None
                    row['modality'] = "text"

                return row
        else:
            key_fields = ["video_filename"]
            mm_field = 'video_filename'
            def func(row):
                raw_frames = row["positive_frames"] + row["negative_frames"]
                row['video_filename'] = []
                for file in raw_frames:
                    clip_name = file['output_path'].replace("/", "_").split(".mp4")[0]
                    clip_name = os.path.join("video_frames", clip_name)
                    row['video_filename'].append(clip_name)
                row['modality'] = "video"
                return row
        dataset = dataset.map(func, load_from_cache_file=False)

    elif dataset_name in ['MSR-VTT', 'MSVD']:
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        mm_field = "video_id"
        key_fields = ['video_id']
    elif dataset_name == "MVBench":

        # prompt_template = """Question: {query}\n\nAnswer with the option\'s letter from the given choices directly and only give the best option."""

        from src.data.eval_dataset.mvbench_dataset import subset_meta
        def func(row):
            subset         = row["subset"]
            query_raw      = row["question"]
            video_filename = row["video"]
            cands_raw      = row["candidates"]
            answer_raw     = row["answer"]

            _, cand_text, answer, answer_idx = qa_template(query_raw, cands_raw, answer_raw)
            query_text = format_qa_with_choices(query_raw, cand_text)
            row['query_text'] = query_text
            row['pos_text'] = answer
            row['video_filename'] = os.path.join(subset, video_filename)
            return row
        
        subsets = []
        for subset_name in subset_meta.keys():
            dataset = load_dataset("OpenGVLab/MVBench", subset_name, split="train")
            new_column = [subset_name] * len(dataset)
            dataset = dataset.add_column("subset", new_column)
            subsets.append(dataset)
        dataset = datasets.concatenate_datasets(subsets)

        dataset = dataset.map(func, load_from_cache_file=False)

        mm_field = "video_filename"
        query_text_field = "query_text"
        target_text_field = "pos_text"
        key_fields = ['question', 'subset', 'video']
    
    elif dataset_name == 'NExTQA':

        dataset = load_dataset("lmms-lab/NExTQA", "MC", split="test")
        mm_field = "query_mm_input"
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
            _, cand_text, _, _ = qa_template(query, options, answer)
            query_text = format_qa_with_choices(query, cand_text)
            row['query_text'] = query_text
            row['pos_text'] = f"({chr(answer + ord('A'))}) {options[answer]}"
            row['query_mm_input'] = os.path.join(config['frame_root'], str(video_id))
            return row
    
        dataset = dataset.map(func, load_from_cache_file=False)
        query_text_field = "query_text"
        target_text_field = "pos_text"
    
    elif dataset_name == "SmthSmthV2":

        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        dataset = sample_dataset(dataset, num_sample_per_subset=1000)

        def func(row):
            frame_dir  = os.path.join(config['frame_root'], str(row['video_id']))
            row['query_mm_path'] = frame_dir
            return row

        dataset = dataset.map(func, load_from_cache_file=False)

        mm_field = "query_mm_path"
        key_fields = ['video_id']
        target_text_field = "pos_text"
    elif dataset_name == "VATEX":

        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        mm_field = "videoID"
        key_fields = ['videoID']
    elif dataset_name in ["HMDB51", "UCF101", "Kinetics-700", "Breakfast"]:

        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        mm_field = "video_id"
        key_fields = ['video_id']
        target_text_field = 'pos_text'
    elif dataset_name == "Video-MME":

        dataset = load_dataset("lmms-lab/Video-MME", split="test")
        mm_field = "videoID"
        key_fields = ['question', 'videoID']
        query_text_field = "query_text"
        target_text_field = "target_text"
        def func(row):
            row['query_text'] = format_qa_with_choices(row['question'], row['options'])
            # answer is A, B, C, ..., convert to index 0, 1, 2, ...
            row['target_text'] = row['options'][ord(row['answer']) - ord('A')]
            return row
        dataset = dataset.map(func, load_from_cache_file=False)

    elif dataset_name in VIDORE_QA_RETRIEVAL_DATASETS:
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

    if encode_side == "target" and isinstance(dataset[0][key_fields[0]], list):
        added = set()
        paired_dataset = []
        for row in tqdm(dataset, disable=os.environ.get("RANK", "0") != "0"):
            for i in range(len(dataset[0][key_fields[0]])):
                key = tuple([row[key_fields[x]][i] for x in range(len(key_fields))])
                if key not in added:
                    added.add(key)
                    to_add = {**row}
                    for k,v in to_add.items():
                        if isinstance(v, list):
                            to_add[k] = v[i]
                    paired_dataset.append(to_add)

        dataset = datasets.Dataset.from_list(paired_dataset)
    
    return {
        "dataset": dataset,
        "mm_field": mm_field,
        "query_text_field": query_text_field,
        "target_text_field": target_text_field,
        "key_fields": key_fields,
        "image_dir": image_dir
    }

def get_unprocessed_data(dataset, encode_side, key_fields, folder):
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
    return dataset, descriptions