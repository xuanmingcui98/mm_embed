from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import os, json
import pickle
from torch.jit import isinstance
import random
from src.model_utils import PHI3V, vlm_image_tokens
from src.utils import print_master, print_rank
# from gen_descriptions import prompts_cot_v2, prefix_keys
import re
import torch
from torch.utils.data import Dataset
from .prompts import (task_categories,
                      query_system_prompts_base,
                      query_system_prompts_cl,
                      query_user_prompts_base,
                      query_user_prompts_cl,
                      target_system_prompts_base,
                      target_system_prompts_with_question,
                      query_user_prompts_cot,
                      target_user_prompts,
                      query_system_prompts_clonly,
                      query_user_prompts_clonly,
                      format_target_user_prompt
                      )

def process_fn(qry, subset):
    if subset in {"CIRR"}:
        return qry.replace("<|image_1|>\nGiven an image, find a similar everyday image with the described changes: ", "").strip()
    elif subset in {"FashionIQ"}:
        return qry.replace("<|image_1|>\nFind an image to match the fashion image and style note: ", "").strip()
    elif subset in {"EDIS"}:
        return qry.replace("<|image_1|>\nFind a news image that matches the provided caption: ", "").strip()
    elif subset in {"RefCOCO"}:
        return qry.replace("<|image_1|>\nSelect the portion of the image that follows the language expressions. ", "").strip()
    elif subset in {"Wiki-SS-NQ"}:
        return qry.replace("Find the document image that can answer the given query: ", "").strip()
    elif subset in {"OVEN"}:
        return qry.replace("<|image_1|>\nRetrieve a Wikipedia image-description pair that provides evidence for the question of this image: ", "").strip()
    elif subset in {"RefCOCO-Matching"}:
        return qry.replace("<|image_1|>\nSelect the portion of the image that follows the language expressions: ", "").strip() 
    elif subset in {"Visual7W-Pointing"}:
        return qry.replace("<|image_1|>\nSelect the portion of the image that answers the question ", "").strip()
    elif subset in {"MSCOCO"}:
        return re.search(r'"([^"]*)"', qry).group(1).strip()
    elif subset in task_categories["vqa"]:
        return qry.replace("<|image_1|>\nRepresent the given image with the following question: ", "").strip()
    elif subset in {"VisualNews_t2i"}:
        return qry.replace("Retrieve an image of this news caption. ", "").strip()
    elif subset in {"MSCOCO_t2i"}:
        return qry.replace("Find me an everyday image that matches the given caption: ", "").strip()
    elif subset in {"WebQA"}:
        return qry.replace("<|image_1|>\nFind a Wikipedia image that answers this question: ", "").strip()
    elif subset in {"VisDial"}:
        return qry.replace("Represent the given dialogue about an image, which is used for image retrieval: ", "").strip()
    elif subset in {"N24News"}:
        return qry.replace("<|image_1|>\nRepresent the given news image with the following caption for domain classification: ", "").strip()
    elif subset in task_categories["classification"] or subset in {"NIGHTS", "MSCOCO_i2t", "VisualNews_i2t"}:
        return None
    else:
        raise ValueError(f"Unknown subset: {subset}")

def process_target_text(text, subset):
    if subset in {"WebQA", "OVEN"}:
        text = text.replace("Represent the given Wikipedia image with related text information: ", "")
    elif subset in {"EDIS"}:
        text = text.replace("Represent the given image with related text information: ", "")
    elif subset in {"RefCOCO-Matching"}:
        text = text.replace("Select the portion of the image that follows the language expressions: ", "")
    
    return text.replace("<|image_1|>", "").strip()

def format_description(description, prompt_format="gt_only"):
    if "<think>" in description:
        index = description.find("<think>")
        # if index > 0:
        #     print_master(f"Removing {v[:index]} from {v}")
        #     n_bad += 1
        description = description[index:]
    if prompt_format == "gt_only":
        return "Answer: " + description.split("Answer: ")[-1].strip(". \n")
    else:
        return description.strip(". \n")

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image

def get_prompt_templates(data_args, model_args):
    # TODO: adhoc. if we are loading fro a ckpt that has to be a sft-only ckpt, so we use the sft prompts (base)
    # if data_args.prompt_format == "cot":
    if data_args.descriptions is not None or data_args.description_dir is not None:
        query_user_prompts = query_user_prompts_cot
    else:
        # if (not model_args.do_cl) or (model_args.sft_checkpoint_path is not None):
        #     query_user_prompts = query_user_prompts_base
        # else:
        #     query_user_prompts = query_user_prompts_cl
        query_user_prompts = query_user_prompts_base
    # query_system_prompts = query_system_prompts_base if (not model_args.do_cl) or (model_args.sft_checkpoint_path is not None)  else query_system_prompts_cl
    query_system_prompts = query_system_prompts_base
    # target_system_prompts = target_system_prompts_with_question if data_args.add_question_to_tgt else target_system_prompts_base
    target_system_prompts = target_system_prompts_with_question if data_args.add_question_to_tgt else target_system_prompts_base

    return (query_system_prompts, query_user_prompts, target_system_prompts)


class TrainTextImageDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        train_data = []

        # disable datasets progress bar if multigpu

        if torch.distributed.is_initialized() and torch.distributed.get_rank() > 0:
            from datasets.utils.logging import disable_progress_bar
            disable_progress_bar()

        query_dataset_length = target_dataset_length = 0
        print_master(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
        for subset in data_args.subset_name:

            query_system_prompts, query_user_prompts, target_system_prompts = get_prompt_templates(data_args, model_args)
            
            subset_data = load_dataset(self.data_args.dataset_name, subset, split=data_args.split_name)[0]

            # TODO: just take 1 row for debugging
            subset_data = subset_data.select([0])

            # remove duplicate rows 

            # first convert to pandas df

            subset_data = subset_data.to_pandas()
            subset_data = subset_data.drop_duplicates(subset=['qry', 'qry_image_path'])
            subset_data = datasets.Dataset.from_pandas(subset_data)
            if data_args.descriptions is not None:
                print_master(f"Loading descriptions from {data_args.descriptions}")
                with open(data_args.descriptions, "rb") as f:
                    descriptions = pickle.load(f)
            elif data_args.description_dir is not None:
                
                desc_path = os.path.join(data_args.description_dir, subset, "cot", "query.pkl")

                if not os.path.exists(desc_path):
                    print_master(f"Descriptions file {desc_path} does not exist. Use no gt query text for {subset}.")
                    desc_path = os.path.join("descriptions", subset, "cot", "query.pkl") # hack

                if not os.path.exists(desc_path):

                    print_master(f"Descriptions file {desc_path} does not exist. Skipping SFT for {subset}.")
                    descriptions = None
                    query_system_prompts = query_system_prompts_clonly
                    query_user_prompts = query_user_prompts_clonly
                else:
                    print_master(f"Loading descriptions from {desc_path}")
                    with open(desc_path, "rb") as f:
                        descriptions = pickle.load(f)
            else:
                descriptions = None

            target_descriptions = None
            if data_args.target_description_dir is not None:
                target_desc_path = os.path.join(data_args.target_description_dir, subset, "cot", "target.pkl")
                if os.path.exists(target_desc_path):
                    print_master(f"Loading target descriptions from {target_desc_path}")
                    with open(target_desc_path, "rb") as f:
                        target_descriptions = pickle.load(f)
                
            max_len = 0

            query_descriptions = []
            target_descriptions_column = []

            if descriptions is not None:
                # filter subset where (row['qry'], row['qry_image_path']) is not in the descriptions
                subset_data = subset_data.filter(lambda row: (row['qry'], row['qry_image_path']) in descriptions, num_proc=None)
                print_master(f"Filtered {subset} dataset to {len(subset_data)} rows based on descriptions.")
            def add_description(row):
                if descriptions is not None and (not data_args.no_description_for_text_only or row['qry_image_path']):

                    description = descriptions[(row['qry'], row['qry_image_path'])]
                    description = format_description(description, self.data_args.prompt_format)
                    nonlocal max_len
                    max_len = max(max_len, len(description.split()))
                    # truncate description to < 50 words
                    description = " ".join(description.split()[:data_args.max_desc_len]) 
                    query_descriptions.append(description)
                else:
                    query_descriptions.append(None)

                if target_descriptions is not None and (not data_args.no_description_for_text_only or row['pos_image_path']):
                    target_descriptions_column.append(format_description(target_descriptions[(row['pos_text'], row['pos_image_path'])], self.data_args.prompt_format))
                else:
                    target_descriptions_column.append(None)

                question = process_fn(row['qry'], subset)
                inputs = {}

                if question is not None:
                    inputs["query"] = question

                if data_args.apply_chat_template:
                    qry = query_user_prompts[subset].format(**inputs)
                else:
                    if descriptions is not None:
                        inputs['answer'] = description
                        # qry = "<|image_1|>\n" + query_user_prompts[subset].format(**inputs)
                        qry = query_user_prompts[subset].format(**inputs)
                    else:
                        qry = row['qry']

                if data_args.apply_chat_template_target:
                    row['pos_text'] = process_target_text(row['pos_text'], subset)
                    # row['pos_text'] = format_target(row['pos_text'], subset, question=question if data_args.add_question_to_tgt else None)
                    row['pos_text'] = format_target_user_prompt(subset, 
                                                                query=question, 
                                                                answer=row['pos_text'], 
                                                                with_question=data_args.add_question_to_tgt, 
                                                                with_description=target_descriptions is not None)

                row['qry'] = qry

                return row


            query_system_prompt = query_system_prompts[subset]
            subset_data = subset_data.map(add_description, batched=False, num_proc=None)
            subset_data = subset_data.add_column("sft_target", ["query"] * len(subset_data))
            subset_data = subset_data.add_column("query_description", query_descriptions)
            subset_data = subset_data.add_column("qry_system_prompt", [query_system_prompt] * len(subset_data))
            subset_data = subset_data.add_column("tgt_system_prompt", [target_system_prompts[subset]] * len(subset_data))
            subset_data = subset_data.add_column("target_description", target_descriptions_column)

            if target_descriptions is not None and model_args.do_sft and not model_args.do_cl:
                # add target data as sft target
                # duplicate subset_data with sft_target = 'target'
                target_subset_data = subset_data.map(lambda row: {
                    "sft_target": "target",
                    "qry": row["pos_text"],
                    "qry_image_path": row["pos_image_path"],
                    "query_description": row["target_description"],
                    "qry_system_prompt": target_system_prompts[subset],
                    "tgt_system_prompt": query_system_prompts[subset],
                }, batched=False, num_proc=None)

                target_subset_data = target_subset_data.to_pandas()
                target_subset_data = target_subset_data.drop_duplicates(subset=['pos_text', 'pos_image_path'])
                target_subset_data = datasets.Dataset.from_pandas(target_subset_data)
                subset_data = concatenate_datasets([subset_data, target_subset_data])
                target_dataset_length += len(target_subset_data)

            # print_master(f"Max length of description in {subset}: {max_len}. Truncated to {data_args.max_desc_len} words.")

            # else:
            #     answer_pool = []
            #     for row in subset_data:
            #         answer_pool.append(row['pos_text'])
            #     answer_pool = set(answer_pool)
            #     def add_description(row):
            #         if self.data_args.perturb_gt_rate > 0:
            #             import random
            #             if random.random() < self.data_args.perturb_gt_rate:
            #                 desc = random.choice(list(answer_pool))
            #             else:
            #                 desc = row['pos_text']
            #         row['qry'] = row['qry'].replace("Represent the given image with the following question: ", "Represent the answer of the following question given the image: ") + f"Answer: {desc}.\n"
            #         return row
            #     subset_data = subset_data.map(add_description, batched=False, num_proc=None)
            
            print_master(f"Loaded {subset} dataset")
            # print_master(f"user query: {subset_data[0]['qry']}")
            # print_master(f"system prompt: {subset_data[0]['qry_system_prompt']}")
            # print_master(f"target system prompt: {subset_data[0]['tgt_system_prompt']}")
            # print_master(f"positive user prompt: {subset_data[0]['pos_text']}")
            train_data.append(subset_data)
        self.train_data = concatenate_datasets(train_data)
        print_master(f"Total training data length: {len(self.train_data)}")
        if model_args.do_sft and model_args.do_sft_target:
            print_master(f"Total target data length: {target_dataset_length}")

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def __getitem__(self, data_idx) -> Tuple[str, List[str]]:
        qry_texts, qry_image_paths, pos_texts, pos_image_paths, query_descriptions, qry_system_prompts, tgt_system_prompts, target_descriptions, sft_targets = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"], 
            self.train_data[data_idx]["query_description"],  self.train_data[data_idx].get("qry_system_prompt", None),
            self.train_data[data_idx].get("tgt_system_prompt", None), self.train_data[data_idx].get("target_description", None),
            self.train_data[data_idx]["sft_target"]
        )
        if 'neg_text' in self.train_data.column_names:
            neg_texts, neg_image_paths = self.train_data[data_idx]["neg_text"], self.train_data[data_idx]["neg_image_path"]
        else:
            if isinstance(data_idx, int):
                neg_texts, neg_image_paths = '', None
            else:
                neg_texts, neg_image_paths = [''] * len(data_idx), [] * len(data_idx)
        if isinstance(data_idx, int):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            neg_texts = [neg_texts]
            neg_image_paths = [neg_image_paths]
            query_descriptions = [query_descriptions]
            qry_system_prompts = [qry_system_prompts] 
            tgt_system_prompts = [tgt_system_prompts]
            target_descriptions = [target_descriptions]
            sft_targets = [sft_targets]
        _qry_texts, _qry_images, _pos_texts, _pos_images, _neg_texts, _neg_images, _query_descriptions, _qry_system_prompts, _tgt_system_prompts, _target_descriptions, _sft_targets = [], [], [], [], [], [], [], [], [], [], []
        backbone = self.model_args.model_backbone
        for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, query_description, qry_system_prompt, tgt_system_prompt, target_description, sft_target \
            in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths, neg_texts, neg_image_paths, query_descriptions, qry_system_prompts, tgt_system_prompts, target_descriptions, sft_targets):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2
            if backbone != PHI3V:
                qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                pos_text = pos_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                neg_text = neg_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone]) if neg_text else None
            qry_image = self._get_image(qry_image_path)
            pos_image = self._get_image(pos_image_path)
            neg_image = self._get_image(neg_image_path) if neg_image_path else None
            if (not qry_text and not qry_image) or (not pos_text and not pos_image):
                print("empty inputs")
                continue
            _qry_texts.append(qry_text)
            _qry_images.append(qry_image)
            _pos_texts.append(pos_text)
            _pos_images.append(pos_image)
            _neg_texts.append(neg_text)
            _neg_images.append(neg_image)
            _query_descriptions.append(query_description)
            _qry_system_prompts.append(qry_system_prompt)
            _tgt_system_prompts.append(tgt_system_prompt)
            _target_descriptions.append(target_description)
            _sft_targets.append(sft_target)

        return  {"query_text": _qry_texts, "query_image": _qry_images, "query_image_path": qry_image_paths,
                "pos_text": _pos_texts, "pos_image": _pos_images, "pos_image_path": pos_image_paths,
                "neg_text": _neg_texts, "neg_image": _neg_images, "query_description": _query_descriptions, 
                "qry_system_prompt": _qry_system_prompts, "tgt_system_prompt": _tgt_system_prompts, "target_description": _target_descriptions, "sft_target": _sft_targets}
        


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone
        self.text_field = text_field

        self.eval_data = load_dataset(
            self.data_args.dataset_name,
            subset,
            split=self.data_args.dataset_split,
        )
        self.subset = subset
        self.query_system, self.query_user, self.target_system_prompts = get_prompt_templates(data_args, model_args)

        # if data_args.prompt_format == 'cot' and data_args.prompt_version == 'v1':
        #     self.reasoning_prefix, self.desc_prefix = prefix_keys[subset][model_args.model_name]

        if data_args.descriptions and os.path.exists(data_args.descriptions):

            print_master(f"Loading descriptions from {data_args.descriptions}")
            with open(data_args.descriptions, "rb") as f:
                self.descriptions = pickle.load(f)

        elif data_args.description_dir is not None and not model_args.do_sft:
            desc_path = os.path.join(data_args.description_dir, subset, "cot", "query.pkl")

            # if not os.path.exists(desc_path):
            #     print_rank(f"Descriptions file {desc_path} does not exist. Use base query text for {subset}.")
            #     self.descriptions = None
            #     self.query_user = query_user_prompts_clonly[subset]
            #     self.query_system = query_system_prompts_clonly[subset]
            # else:
            print_master(f"Loading descriptions from {desc_path}")
            with open(desc_path, "rb") as f:
                self.descriptions = pickle.load(f)
        else:

            self.descriptions = None
        # else:
        #     if text_field == "qry_text":
        #         tgt_field_name = "tgt_text" if "tgt_text" in self.eval_data.column_names else "pos_text"
        #         self.answers = {}
        #         for row in self.eval_data:
        #             self.answers[(row[text_field], row[img_path_field])] = row[tgt_field_name][0]
        #         self.answer_pool = set(self.answers.values())

        self.target_descriptions = None
        if data_args.target_description_dir is not None and not model_args.do_sft_target:
            target_desc_path = os.path.join(data_args.target_description_dir, subset, "cot", "target.pkl")
            if os.path.exists(target_desc_path):
                print_master(f"Loading target descriptions from {target_desc_path}")
                with open(target_desc_path, "rb") as f:
                    self.target_descriptions = pickle.load(f)
            else:
                print_master(f"Target descriptions file {target_desc_path} does not exist. Ignored.")
                
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data],
            "query": [pair["query"] for pair in self.paired_data]
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path, question = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"], self.paired_dataset[item].get("query", None)
        orig_text = text
        description = qry_system_prompt = tgt_system_prompt = None
        # if os.path.exists(self.data_args.descriptions):
        if self.text_field == "qry_text":
            if self.descriptions and (not self.data_args.no_description_for_text_only or img_path):
                description = self.descriptions[(text, img_path)]
                description = format_description(description, self.data_args.prompt_format)
                # description = " ".join(description.split()[:self.data_args.max_desc_len])  # truncate description 

            question = process_fn(text, self.subset)
            inputs = {}
            
            if question is not None:
                inputs['query'] = question
            
            if self.data_args.apply_chat_template:
                text = self.query_user[self.subset].format(**inputs)
                qry_system_prompt = self.query_system[self.subset]

            # if description:
            #     description = format_description(description, self.data_args.prompt_format)# TODO: change this

        else:
            description = format_description(self.target_descriptions[(text, img_path)], self.data_args.prompt_format) if self.target_descriptions and (not self.data_args.no_description_for_text_only or img_path) else None
            if self.data_args.apply_chat_template_target:
                text = process_target_text(text, self.subset)
                if question is not None:
                    question = process_fn(question, self.subset)
                    
                text = format_target_user_prompt(self.subset, query=question, answer=text, with_question=self.data_args.add_question_to_tgt, with_description=self.target_descriptions is not None)
                tgt_system_prompt = self.target_system_prompts[self.subset]
            else:
                raise NotImplementedError("Target chat template is required.")
                

            # else:
            #     # for vqa with GT answers, we may not need a description file
            #     if self.subset in {"InfographicsVQA"} and self.text_field == "qry_text":
            #         if random.random() < self.data_args.perturb_gt_rate:
            #             desc = random.choice(list(self.answer_pool))
            #         else:
            #             desc = self.answers[(text, img_path)]
            #         text = text.replace("Represent the given image with the following question: ", "Represent the answer of the following question given the image: ") + f"Answer: {desc}.\n"

        if self.backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.backbone])

        return text, self._get_image(img_path), img_path, qry_system_prompt, tgt_system_prompt, description, orig_text

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field], None))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path, None))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field], None))
            elif type(row[text_field]) == list:
                assert type(row[img_path_field]) == list and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path, None)) # row['qry_text'] if (self.subset in task_categories["vqa"].union(task_categories['classification'])) and self.data_args.apply_chat_template_target else None))

        paired_data = [{"text": text, "img_path": img_path, "query": query} for text, img_path, query in unique_pair]
        # sort to a unique order
        paired_data.sort(key=lambda x: (x["text"], x["img_path"]))
        return paired_data


class FlickrDataset(Dataset):
    def __init__(self, modality, model_backbone):
        self.model_backbone = model_backbone
        self.modality = modality
        self.raw_data = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval", split="test")
        if modality == "image":
            self.eval_data, self.image_names = self.get_image_data()
        else:
            self.eval_data, self.image_names = self.get_text_data()

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        text, image = self.eval_data[idx]
        if self.model_backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_backbone])
            if self.data_args.image_resolution:
                image = process_image(image, self.data_args.image_resolution)
        return text, image

    def get_image_data(self):
        eval_data, image_names = [], []
        # i2t
        inst = "<|image_1|> Find an image caption describing the given image."
        # inst = "<|image_1|> Represent the given image for image caption retrieval."
        # t2i
        # inst = "<|image_1|> Represent the given image."

        for row in self.raw_data:
            eval_data.append((inst, row["image"]))
            image_names.append(row["filename"])
        return eval_data, image_names

    def get_text_data(self):
        eval_data, image_names = [], []
        # i2t
        inst = ""
        # t2i
        # inst = "Retrieve an image that matches the given caption: "
        # inst = "Find me an everyday image that matches the given caption."  # MSCOCO t2i
        for row in self.raw_data:
            for caption in row["caption"]:
                # eval_data.append((caption, None))
                eval_data.append((inst + caption, None))
                image_names.append(row["filename"])
        return eval_data, image_names
