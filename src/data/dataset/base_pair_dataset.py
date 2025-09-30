from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from functools import partial
from datasets import Features, Value, Sequence
import pickle, os
from datasets import load_dataset, concatenate_datasets
from ...model.processor import PHI3V, VLM_IMAGE_TOKENS, VLM_VIDEO_TOKENS
from ...utils import print_master, print_rank
from ..prompts import (format_description,
                       extract_query, extract_target,
                       query_user_prompts_cot, target_user_prompts_cot,
                       IMAGE_TASKS,
                       query_user_prompts_cot_generation, target_user_prompts_cot_generation,
                       TASK2ID)
from ..utils.dataset_utils import sample_dataset
from ..utils.vision_utils import save_frames, load_frames, sample_frames
import torch
from ..loader.mixed_dataset import add_metainfo_hook
import torch
import json
import numpy as np

DATASET_INSTRUCTION = {
    'Kinetics-700': 'Recognize the category of the video content.',
    'SmthSmthV2': 'What actions or object interactions are being performed by the person in the video?',
    'UCF101': 'What activities or sports are being performed by the person in the video?',
    'HMDB51': 'What actions or objects interactions are the person in the video doing?',
    'Breakfast': 'Recognize the breakfast type that the person is cooking in the video. ',
}

# MULTIMODAL_FEATURES = Features(**{
#     "query_text": Value(dtype='string', id=None),
#     "query_image": {'bytes': Value(dtype='null', id=None), 'path': Value(dtype='null', id=None)},
#     "pos_text": Value(dtype='string', id=None),
#     "pos_image": {'bytes': Value(dtype='null', id=None), 'path': Value(dtype='null', id=None)},
#     "neg_text": Value(dtype='string', id=None),
#     "neg_image": {'bytes': Value(dtype='null', id=None), 'path': Value(dtype='null', id=None)},
#     "global_dataset_name": Value(dtype='string', id=None),
# })


MULTIMODAL_FEATURES = Features(**{
    "query_text": Value(dtype='string'),
    "query_image": {
        "paths": Sequence(Value(dtype='string')),  # List of image paths (frames)
        "bytes": Sequence(Value(dtype='binary')),  # List of pre-saved image bytes
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))  # List of [width, height] pairs
    },
    "pos_text": Value(dtype='string'),
    "pos_image": {
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    },
    "neg_text": Value(dtype='string'),
    "neg_image": {
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    },
    "global_dataset_name": Value(dtype='string'),
    "task_id": Value(dtype='int32')  # Task ID for the dataset
})

RESOLUTION_MAPPING = {
    "high": (1344, 1344),
    "mid": (672, 672),
    "low": (128, 128),
}


class BaseDatasetProcessor:
    def __init__(self,
                 data_parser_name,
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 instruction=None,
                 **dataset_config):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.processor = processor
        self.dataset_config = dataset_config
        self.data_parser_name = data_parser_name

        self.dataset_name = self.dataset_config.get("dataset_name")
        self.subset_name = self.dataset_config.get("subset_name", self.dataset_name)
        self.dataset_split = self.dataset_config.get("dataset_split", "original")
        self.image_dir = self.dataset_config.get('image_dir')
        self.model_backbone = self.model_args.model_backbone
        self.image_resolution = self.dataset_config.get('image_resolution')
        self.instruction = instruction
        self.encode_side = self.dataset_config.get('encode_side', 'query')

        self.query_descriptions = self.target_descriptions = None
        if data_args.query_description_dir is not None:
            desc_path = os.path.join(data_args.query_description_dir, self.subset_name, "cot", "query.pkl")
            if os.path.exists(desc_path):
                with open(desc_path, "rb") as f:
                    self.query_descriptions = pickle.load(f)

        if data_args.target_description_dir is not None:
            desc_path = os.path.join(data_args.target_description_dir, self.subset_name, "cot", "target.pkl")
            if os.path.exists(desc_path):
                with open(desc_path, "rb") as f:
                    self.target_descriptions = pickle.load(f)

        if self.model_args.meta_queries is not None and self.model_args.meta_queries > 0:
            self.meta_queries = "".join(
                [f'<meta_query_{i}>' for i in range(self.model_args.meta_queries)]
            )
        else:
            self.meta_queries = ''

        self.dataset_config['global_dataset_name'] = f'{self.data_parser_name}/{self.subset_name}'
        self.dataset_config['model_backbone'] = self.model_args.model_backbone

        self.dataset = self._load_hf_dataset()
        # self.add_signature_columns()
        self.clusters = os.path.join(data_args.cluster_path, f"{self.subset_name}_clusters.json")
        if os.path.exists(self.clusters):
            all_clusters = []
            if self.clusters.endswith('json'):
                with open(self.clusters, 'r') as file:
                    clusters = json.load(file)
            elif self.clusters.endswith('jsonl'):
                with open(self.clusters, 'r') as file:
                    clusters = [json.loads(line) for line in file]
            for c in clusters:
                all_clusters.append(c['cluster'])
            if len(all_clusters) < len(self.dataset):
                all_clusters += [-1] * (len(self.dataset) - len(all_clusters))
            cluster_idx = np.argsort(all_clusters)
            self.dataset = self.dataset.select(cluster_idx)

        if self.data_args.debug_prompt:
            print_master(f"Debug mode enabled")
            self.dataset = self.dataset.select(range(1)) 

        self.column_names = self.dataset.column_names
        self.dataset = sample_dataset(self.dataset, **dataset_config)


    def load(self):
        num_rows = self.dataset.num_rows
        self.dataset_config['image_resolution'] = self.data_args.image_resolution

        columns_to_remove = []
        for col_name in self.column_names:
            if col_name not in MULTIMODAL_FEATURES:
                columns_to_remove.append(col_name)

        num_rows = self.dataset.num_rows
        n_workers_per_node = self.training_args.dataloader_num_workers if self.training_args.dataloader_num_workers > 0 else 1
        # if self.data_args.combine_image_datasets:
        #     n_workers_per_node = 8 * self.dataset_config['world_size'] * n_workers_per_node

        if not self.data_args.debug_prompt:
            self.dataset = self.dataset.to_iterable_dataset(num_shards=8)
            setattr(self.dataset, 'num_rows', num_rows)

        self.dataset = self.dataset.map(
                            lambda x:
                            self.batch_preprocess(x, data_args=self.data_args, model_args=self.model_args, processor=self.processor, **self.dataset_config),
                            batched=True, 
                            batch_size=64,
                            remove_columns=columns_to_remove,
                            drop_last_batch=not self.data_args.debug_prompt # temp
                            )

        self.dataset = self.dataset.cast(MULTIMODAL_FEATURES)

        print_master(f"Loaded {self.data_parser_name}/{self.dataset_name} dataset with {num_rows} samples")
        if not self.data_args.debug_prompt:

            setattr(self.dataset, 'num_rows', num_rows)
        
        return self.dataset

    def _load_hf_dataset(self):
        """
            Load the dataset based on the configuration.
            May be implemented by subclasses.
        """

        dataset = load_dataset(self.dataset_name, self.dataset_config['subset_name'], split=f"{self.dataset_split}")

        return dataset


    def _process_one_sample(self, idx, batch_dict, *args, **kwargs):
        """
            Process one sample in the batch.
            May be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def format_text_for_chat_template(self, is_query, text, image_path=None, video_path=None, description=None, add_generation_prompt=False):

        if is_query:
            extract_fn = extract_query
            # instruction = self.instruction['query'] if self.instruction is not None else None
            instruction = query_user_prompts_cot[self.subset_name]
        else:
            extract_fn = extract_target
            # instruction = self.instruction['target'] if self.instruction is not None else None
            instruction = target_user_prompts_cot[self.subset_name]

        text = extract_fn(text, self.subset_name)
        # make sure no extra visual tokens are left in the text
        for img_tok in VLM_IMAGE_TOKENS.values():
            text = text.replace(img_tok, "")
        for vid_tok in VLM_VIDEO_TOKENS.values():
            text = text.replace(vid_tok, "")

        # if instruction is not None:
        #     text = instruction.format(text=text)

        text = instruction.format(query=text)

        description = format_description(description, self.data_args.prompt_format)

        if self.data_args.max_rewrite_len:
            description = " ".join(description.split()[:self.data_args.max_rewrite_len])

        formatted_sample = [
            {"role": "system",
            "content": "You are a helpful assistant specialized in multimodal embedding."}
        ]
        
        user_content = [] 
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        if video_path:
            user_content.append({"type": "video", "video": video_path})
        user_content.append({"type": "text", "text": text})
        formatted_sample.append({"role": "user", "content": user_content})

        if not add_generation_prompt:
            formatted_sample.append({
                "role": "assistant",
                "content": [{"type": "text", "text": description}],
            })
        
        formatted_sample = self.processor.apply_chat_template(formatted_sample, add_generation_prompt=add_generation_prompt, tokenize=False)
        if not add_generation_prompt:
            formatted_sample = formatted_sample.strip()

        if self.meta_queries:
            formatted_sample += self.meta_queries
        return formatted_sample 


    @add_metainfo_hook
    def batch_preprocess(self, batch_dict, data_args, model_args, processor, *args, **kwargs):

        batch_size = len(batch_dict['qry'])
        query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
        for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path in \
            zip(batch_dict['qry'], batch_dict['qry_image_path'],
                batch_dict['pos_text'], batch_dict['pos_image_path'],
                batch_dict.get('neg_text', [''] * batch_size), batch_dict.get('neg_image_path', [None] * batch_size)):
            if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
                print(f"empty inputs from {self.subset_name}")
                continue

            if data_args.apply_chat_template:
                query_description = pos_description = None
                if self.query_descriptions:
                    query_description = self.query_descriptions[(qry_text, qry_image_path)]
                if self.target_descriptions:
                    pos_description = self.target_descriptions[(pos_text, pos_image_path)]

                if (not qry_image_path and self.data_args.rewrites_for_mm_only): # or self.encode_side == 'target':
                    query_description = None
                if (not pos_image_path and self.data_args.rewrites_for_mm_only): # or self.encode_side == 'query':
                    pos_description = None
                qry_text = self.format_text_for_chat_template(is_query=True, text=qry_text, image_path=qry_image_path, add_generation_prompt=False, description=query_description)
                pos_text = self.format_text_for_chat_template(is_query=False, text=pos_text, image_path=pos_image_path, add_generation_prompt=False, description=pos_description)
            else:
                if self.model_backbone != PHI3V:
                    qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.model_backbone])
                    pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.model_backbone])
                    neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.model_backbone]) if neg_text else ''

            query_texts.append(qry_text)
            pos_texts.append(pos_text)
            neg_texts.append(neg_text)
            # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
            qry_image = {"bytes": [None], "paths": [os.path.join(self.image_dir, qry_image_path) if qry_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(self.image_resolution, None)]}
            pos_image = {"bytes": [None], "paths": [os.path.join(self.image_dir, pos_image_path) if pos_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(self.image_resolution, None)]}
            neg_image = {"bytes": [None], "paths": [os.path.join(self.image_dir, neg_image_path) if neg_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(self.image_resolution, None)]}
            query_images.append(qry_image)
            pos_images.append(pos_image)
            neg_images.append(neg_image)
        if len(query_texts) == 0:
            print('something went wrong')
        # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
        return {"query_text": query_texts, "query_image": query_images,
                "pos_text": pos_texts, "pos_image": pos_images,
                "neg_text": neg_texts, "neg_image": neg_images}

    @add_metainfo_hook
    def single_preprocess(self, sample, data_args, model_args, processor, *args, **kwargs):
        qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path = \
            sample['qry'], sample['qry_image_path'], \
            sample['pos_text'], sample['pos_image_path'], \
            sample.get('neg_text', ''), sample.get('neg_image_path', None)
        if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
            print("empty inputs")
            return None
        
        if data_args.apply_chat_template:
            query_description = pos_description = None
            if self.query_descriptions:
                query_description = self.query_descriptions[(qry_text, qry_image_path)]
            if self.target_descriptions:
                pos_description = self.target_descriptions[(pos_text, pos_image_path)]

            if not qry_image_path and self.data_args.rewrites_for_mm_only:
                query_description = None
            if not pos_image_path and self.data_args.rewrites_for_mm_only:
                pos_description = None
            qry_text = self.format_text_for_chat_template(is_query=True, text=qry_text, image_path=qry_image_path, add_generation_prompt=False, description=query_description)
            pos_text = self.format_text_for_chat_template(is_query=False, text=pos_text, image_path=pos_image_path, add_generation_prompt=False, description=pos_description)
        else:
            if self.model_backbone != PHI3V:
                qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.model_backbone])
                pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.model_backbone])
                neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.model_backbone]) if neg_text else ''
        qry_image = {"bytes": [None], "paths": [os.path.join(self.image_dir, qry_image_path) if qry_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(self.image_resolution, None)]}
        pos_image = {"bytes": [None], "paths": [os.path.join(self.image_dir, pos_image_path) if pos_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(self.image_resolution, None)]}
        neg_image = {"bytes": [None], "paths": [os.path.join(self.image_dir, neg_image_path) if neg_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(self.image_resolution, None)]}
        return {"query_text": qry_text, "query_image": qry_image,
                "pos_text": pos_text, "pos_image": pos_image,
                "neg_text": neg_text, "neg_image": neg_image}
    

class VideoDatasetProcessor(BaseDatasetProcessor):
    def __init__(self, dataset_parser_name, model_args, data_args, training_args, **kwargs):
        super().__init__(dataset_parser_name, model_args, data_args, training_args, **kwargs)

    def _load_hf_dataset(self):
        """
            Load the dataset based on the configuration.
            May be implemented by subclasses.
        """

        return load_dataset(self.dataset_name, split="train")


    @add_metainfo_hook
    def batch_preprocess(self, batch_dict, *args, **kwargs):

        query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
        batch_size = len(next(iter(batch_dict.values())))

        for data_idx in range(batch_size):
            one_sample = self._process_one_sample(data_idx, batch_dict, *args, **kwargs)
            query_text, query_image, pos_text, pos_image, neg_text, neg_image, query_description, target_description  = \
                one_sample['query_text'], one_sample['query_image'], \
                one_sample['pos_text'], one_sample['pos_image'], \
                one_sample['neg_text'], one_sample['neg_image'], \
                one_sample['query_description'], one_sample['target_description']

            if self.data_args.apply_chat_template:

                query_image_input = query_video_input = None
                if query_image:
                    if len(query_image['paths']) > 1:
                        query_video_input = query_image['paths'][0] or query_image['bytes'][0]
                    else:
                        query_image_input = query_image['paths'][0] or query_image['bytes'][0]
                
                pos_image_input = pos_video_input = None
                if pos_image:
                    if len(pos_image['paths']) > 1:
                        pos_video_input = pos_image['paths'][0] or pos_image['bytes'][0]
                    else:
                        pos_image_input = pos_image['paths'][0] or pos_image['bytes'][0]
                
                if (not query_image and self.data_args.rewrites_for_mm_only): # or self.encode_side == 'target':
                    query_description = None
                if (not pos_image and self.data_args.rewrites_for_mm_only): # or self.encode_side == 'query':
                    target_description = None
                query_text = self.format_text_for_chat_template(
                    is_query=True, text=query_text, image_path=query_image_input, video_path=query_video_input, 
                    description=query_description)
                pos_text = self.format_text_for_chat_template(
                    is_query=False, text=pos_text, image_path=pos_image_input, video_path=pos_video_input, 
                    description=target_description)

            query_texts.append(query_text)
            query_images.append(query_image)
            pos_texts.append(pos_text)
            pos_images.append(pos_image)
            neg_texts.append(neg_text)
            neg_images.append(neg_image)

        return {"query_text": query_texts, "query_image": query_images,
                "pos_text": pos_texts, "pos_image": pos_images,
                "neg_text": neg_texts, "neg_image": neg_images,}


    def format_text_for_chat_template(self, is_query, text, image_path=None, video_path=None, description=None, add_generation_prompt=False):

        if is_query:
            extract_fn = extract_query
            instruction = self.instruction['query'] if self.instruction is not None else None
        else:
            extract_fn = extract_target
            instruction = self.instruction['target'] if self.instruction is not None else None

        text = extract_fn(text, self.subset_name)
        # make sure no extra visual tokens are left in the text
        for img_tok in VLM_IMAGE_TOKENS.values():
            text = text.replace(img_tok, "")
        for vid_tok in VLM_VIDEO_TOKENS.values():
            text = text.replace(vid_tok, "")

        # if instruction is not None:
        #     text = instruction.format(text=text)

        description = format_description(description, self.data_args.prompt_format)

        if self.data_args.max_rewrite_len:
            description = " ".join(description.split()[:self.data_args.max_rewrite_len])

        formatted_sample = [
            {"role": "system",
            "content": "You are a helpful assistant specialized in multimodal embedding."}
        ]
        
        user_content = [] 
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        if video_path:
            user_content.append({"type": "video", "video": video_path})
        user_content.append({"type": "text", "text": text})
        formatted_sample.append({"role": "user", "content": user_content})

        if not add_generation_prompt:
            formatted_sample.append({
                "role": "assistant",
                "content": [{"type": "text", "text": description}],
            })
        
        formatted_sample = self.processor.apply_chat_template(formatted_sample, add_generation_prompt=add_generation_prompt, tokenize=False)
        if not add_generation_prompt:
            formatted_sample = formatted_sample.strip()

        if self.meta_queries:
            formatted_sample += self.meta_queries
        return formatted_sample 

SFT_FEATURES = Features(**{
    "prompt": Value(dtype='string'),
    "image": {
        "paths": Sequence(Value(dtype='string')),  # List of image paths (frames)
        "bytes": Sequence(Value(dtype='binary')),  # List of pre-saved image bytes
    },
    "answer": Value(dtype='string'),
})



class BaseSFTDatasetProcessor:
    def __init__(self,
                 data_parser_name,
                 model_args, 
                 data_args, 
                 training_args, 
                 processor,
                 **dataset_config):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.processor = processor
        self.dataset_config = dataset_config
        self.data_parser_name = data_parser_name

        self.dataset_name = self.dataset_config.get("subset_name", self.dataset_config.get("dataset_name", None))
        self.dataset_split = self.dataset_config.get("dataset_split", "original")
        self.image_dir = self.dataset_config.get('image_dir', None)
        self.model_backbone = self.model_args.model_backbone
        self.image_resolution = self.dataset_config.get('image_resolution', None)
        self.encode_side = self.dataset_config.get('encode_side', 'query')

        self.query_descriptions = self.target_descriptions = None

        self.dataset_config['global_dataset_name'] = f'{self.data_parser_name}/{self.dataset_name}'
        self.dataset_config['model_backbone'] = self.model_args.model_backbone
        self.dataset_config['subset_name'] = self.dataset_name

        self.dataset = self._load_hf_dataset()

        if self.data_args.debug_prompt:
            print_master(f"Debug mode enabled")
            self.dataset = self.dataset.select(range(1)) 

        self.column_names = self.dataset.column_names
        self.dataset = sample_dataset(self.dataset, **dataset_config)

        if self.encode_side == 'query':
            desc_path = os.path.join(data_args.query_description_dir, self.dataset_name, "cot", "query.pkl")
            if os.path.exists(desc_path):
                with open(desc_path, "rb") as f:
                    self.query_descriptions = pickle.load(f)
            self.prompt_template = query_user_prompts_cot_generation[self.dataset_name]
        else:
            desc_path = os.path.join(data_args.target_description_dir, self.dataset_name, "cot", "target.pkl")
            if os.path.exists(desc_path):
                with open(desc_path, "rb") as f:
                    self.target_descriptions = pickle.load(f)
            self.prompt_template = target_user_prompts_cot_generation[self.dataset_name]

    def load(self):
        num_rows = self.dataset.num_rows
        self.dataset_config['image_resolution'] = self.data_args.image_resolution
        columns_to_remove = []
        for col_name in self.column_names:
            if col_name not in {"image_path", "image", "prompt", "answer"}:
                columns_to_remove.append(col_name)
        self.dataset = self.dataset.map(
                            lambda x:
                            self.batch_preprocess(x, data_args=self.data_args, model_args=self.model_args, processor=self.processor, **self.dataset_config),
                            batched=True, 
                            batch_size=1024,
                            num_proc=os.cpu_count() // 2 if not self.data_args.debug_prompt else None,
                            remove_columns=columns_to_remove,
                            )
        self.dataset = self.dataset.cast(SFT_FEATURES)

        print_master(f"Loaded {self.data_parser_name}/{self.dataset_name} dataset with {num_rows} samples")
        
        return self.dataset

    def _load_hf_dataset(self):
        """
            Load the dataset based on the configuration.
            May be implemented by subclasses.
        """

        dataset = load_dataset("MMEB-train", self.dataset_name, split=f"{self.dataset_split}")

        return dataset


    def _process_one_sample(self, idx, batch_dict, *args, **kwargs):
        """
            Process one sample in the batch.
            May be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def batch_preprocess(self, batch_dict, model_args, processor, *args, **kwargs):

        texts, images, answers = [], [], []
        for qry_text, qry_image_path, pos_text, pos_image_path in \
            zip(batch_dict['qry'], batch_dict['qry_image_path'],
                batch_dict['pos_text'], batch_dict['pos_image_path']):
            if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
                print("empty inputs")
                continue
            
            if self.encode_side == 'query':
                text = self.prompt_template.format(query=extract_query(qry_text, self.dataset_name))
                image_path = qry_image_path
                answer = self.query_descriptions[(qry_text, qry_image_path)]
            else:
                text = self.prompt_template.format(query=extract_target(pos_text, self.dataset_name))
                image_path = pos_image_path
                answer = self.target_descriptions[(pos_text, pos_image_path)]

            for img_tok in VLM_IMAGE_TOKENS.values():
                text = text.replace(img_tok, "")
            for vid_tok in VLM_VIDEO_TOKENS.values():
                text = text.replace(vid_tok, "")

            image = {"bytes": [None], "paths": [os.path.join(self.image_dir, image_path)]}

            texts.append(text)
            answers.append(answer)
            images.append(image)

        return {"prompt": texts, 
                "image": images, 
                "answer": answers}

class VideoSFTDatasetProcessor(BaseSFTDatasetProcessor):
    def __init__(self, dataset_parser_name, model_args, data_args, training_args, **kwargs):
        super().__init__(dataset_parser_name, model_args, data_args, training_args, **kwargs)

    def batch_preprocess(self, batch_dict, model_args, processor, *args, **kwargs):

        prompts, images, answers = [], [], []
        batch_size = len(next(iter(batch_dict.values())))
        for data_idx in range(batch_size):
            one_sample = self._process_one_sample(data_idx, batch_dict, *args, **kwargs)
            prompt, image, answer = one_sample['prompt'], one_sample['image'], one_sample['answer']
            if not answer:
                continue

            for img_tok in VLM_IMAGE_TOKENS.values():
                prompt = prompt.replace(img_tok, "")
            for vid_tok in VLM_VIDEO_TOKENS.values():
                prompt = prompt.replace(vid_tok, "")

            prompt = self.prompt_template.format(query=prompt)

            prompts.append(prompt)
            images.append(image)
            answers.append(answer)

        return {"prompt": prompts, "answer": answers, "image": images}