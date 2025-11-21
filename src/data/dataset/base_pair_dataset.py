from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from functools import partial
from datasets import Features, Value, Sequence
import random
import pickle, os
from datasets import load_dataset, concatenate_datasets, load_from_disk
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
import copy
from ...utils import tqdm0

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
    "neg_text": Sequence(Value(dtype='string')),
    "neg_image": Sequence({
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    }),
    "global_dataset_name": Value(dtype='string'),
    "task_id": Value(dtype='int32'),  # Task ID for the dataset
    "index_id": Value(dtype="int32"),
    "query_ecr": Value(dtype="string"),
    "pos_ecr": Value(dtype="string"),
    # "neg_scores": Sequence(Value(dtype="float"))
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
        self.dataset = self.dataset.add_column("index_id", list(range(len(self.dataset))))

        

        if self.data_args.hard_negative_dir:
            self.non_iter_dataset = copy.deepcopy(self.dataset)
            
            # self.dataset = self.dataset.select(list(range(2048)))
            cache_location = f"cache_dir/{self.subset_name}_{self.data_args.hard_negative_filter_strategy}_{self.data_args.hard_negatives_per_sample}_threshold{self.data_args.hard_negative_filter_threshold}"
            if os.path.exists(cache_location):
                self.dataset = load_from_disk(cache_location)
            else:
                hard_negatives_dict = pickle.load(open(os.path.join(self.data_args.hard_negative_dir, f"{self.subset_name}.pkl"), "rb"))

                global MULTIMODAL_FEATURES
                MULTIMODAL_FEATURES["hard_negatives"] = Sequence(Value("int32"))

                print_master("==> Loading hard negatives")
                hard_negatives = []
                has_sufficient_hn = []
                for row in tqdm0(self.dataset):
                    anno = hard_negatives_dict.get(row['index_id'])
                    if not anno:
                        has_sufficient_hn.append(False)
                        hard_negatives.append([]) # dummy
                        continue

                    scores, candidate_ids, gt_id = \
                        anno['scores'], anno['candidate_ids'], anno['ground_truth_id']
                    if self.data_args.hard_negative_filter_strategy == "random":
                        no_gt = [x for x in candidate_ids if x != gt_id]
                        if len(no_gt) < self.data_args.hard_negatives_per_sample:
                            hard_negative = [] # dummy
                        else:
                            hard_negative = random.sample(no_gt, self.data_args.hard_negatives_per_sample)
                    else:
                        gt_score = scores[candidate_ids.index(gt_id)]
                        if self.data_args.hard_negative_filter_strategy == "pp":
                            threshold = self.data_args.hard_negative_filter_threshold * gt_score
                            zipped = list(zip(scores, candidate_ids))
                            zipped = sorted(zipped, lambda x: -x[0])
                            filtered = [x for x in zipped if x[0] < threshold]
                            hard_negative = filtered[:self.data_args.hard_negatives_per_sample]
                        else:
                            raise NotImplementedError()
                    
                    has_sufficient_hn.append(len(hard_negative) == self.data_args.hard_negatives_per_sample)
                    hard_negatives.append(hard_negative)
                self.dataset = self.dataset.add_column("hard_negatives", hard_negatives)
                indices = np.where(has_sufficient_hn)[0].tolist()
                self.dataset = self.dataset.select(indices)
                print_master(f"Filtered out {len(has_sufficient_hn) - sum(has_sufficient_hn)} rows due to insufficinet Hard Negative samples.")
                self.dataset.save_to_disk(cache_location)


        # self.add_signature_columns()
        if self.subset_name == "video_qa_240k":
            cluster_filename = "video_240k_caption_15k_clusters.jsonl"
        elif self.subset_name in  {"video_caption_300k-v2t", "video_caption_300k-t2v"}:
            cluster_filename = "video_caption_300k_clusters.jsonl"
        elif self.subset_name == "VisRAG-Indomain-data":
            cluster_filename = "VisRAG-Ret-Train-In-domain-data_clusters.json"
        else:
            cluster_filename = f"{self.subset_name}_clusters.json"
        self.clusters = os.path.join(data_args.cluster_path, cluster_filename)
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
            
            self.dataset = self.dataset.add_column("cluster", all_clusters)
            cluster_idx = np.argsort(all_clusters)
            self.dataset = self.dataset.select(cluster_idx)

            MULTIMODAL_FEATURES["cluster"] = Value(dtype='int32')

        if self.data_args.debug_prompt:
            print_master(f"Debug mode enabled")
            # self.dataset = self.dataset.select(range(1024)) 

        self.column_names = self.dataset.column_names
        self.dataset = sample_dataset(self.dataset, **dataset_config)


    def load(self):
        num_rows = self.dataset.num_rows
        self.dataset_config['image_resolution'] = self.data_args.image_resolution

        columns_to_remove = []
        for col_name in self.column_names:
            if col_name not in MULTIMODAL_FEATURES:
                columns_to_remove.append(col_name)

        n_workers_per_node = self.training_args.dataloader_num_workers if self.training_args.dataloader_num_workers > 0 else 1
        # if self.data_args.combine_image_datasets:
        #     n_workers_per_node = 8 * self.dataset_config['world_size'] * n_workers_per_node

        if self.data_args.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.training_args.seed)

        if not self.data_args.debug_prompt:
            self.dataset = self.dataset.to_iterable_dataset(num_shards=8)
            setattr(self.dataset, 'num_rows', num_rows)

        self.dataset = self.dataset.map(
                            lambda x:
                            self.batch_preprocess(x, data_args=self.data_args, model_args=self.model_args, processor=self.processor, **self.dataset_config),
                            batched=True, 
                            batch_size=64,
                            remove_columns=columns_to_remove,
                            drop_last_batch=False #not self.data_args.debug_prompt # temp
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

        # # filter
        # filtered = dataset.filter(
        #     lambda row: (row["qry"] or row["qry_image_path"]) and 
        #                 (row["pos_text"] or row["pos_image_path"])
        # )

        # print_master(f"Loading {self.subset_name} dataset. Original length: {len(dataset)}. After filter: {len(filtered)}. Filtered out {len(dataset) - len(filtered)} samples.")

        # return filtered
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
        query_ecrs, pos_ecrs = [], []
        neg_scores = []
        for idx, (qry_text, qry_image_path, pos_text, pos_image_path) in \
            enumerate(zip(batch_dict['qry'], batch_dict['qry_image_path'],
                batch_dict['pos_text'], batch_dict['pos_image_path'])):
            if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
                print(f"empty inputs from {self.subset_name}")
                continue

            hard_negative_texts = [''] * self.data_args.hard_negatives_per_sample
            hard_negative_images = [''] * self.data_args.hard_negatives_per_sample
            hard_negative_description = [''] * self.data_args.hard_negatives_per_sample

            if self.data_args.hard_negative_dir:
                hard_negative_samples = batch_dict['hard_negatives'][idx]
                hard_negative_samples = [self.non_iter_dataset[i] for i in hard_negative_samples]
                hard_negative_texts = [x['pos_text'] for x in hard_negative_samples]
                hard_negative_images = [x['pos_image_path'] for x in hard_negative_samples]
                

            query_description = pos_description = None
            if self.query_descriptions:
                query_description = self.query_descriptions[(qry_text, qry_image_path)]

            if self.target_descriptions:
                pos_description = self.target_descriptions[(pos_text, pos_image_path)]
                if self.data_args.hard_negative_dir:
                    hard_negative_description = [self.target_descriptions[(t, i)] for t,i in zip(hard_negative_texts, hard_negative_images)]

            # if (not qry_image_path and self.data_args.rewrites_for_mm_only) or self.encode_side == 'target':
            #     query_description = None
            # if (not pos_image_path and self.data_args.rewrites_for_mm_only) or self.encode_side == 'query':
            #     pos_description = None
            #     hard_negative_description = [None] * len(hard_negative_description)

            if data_args.apply_chat_template:
                qry_text = self.format_text_for_chat_template(is_query=True, text=qry_text, image_path=qry_image_path, add_generation_prompt=False, description=query_description)
                pos_text = self.format_text_for_chat_template(is_query=False, text=pos_text, image_path=pos_image_path, add_generation_prompt=False, description=pos_description)
                if self.data_args.hard_negative_dir:
                    hard_negative_texts = [self.format_text_for_chat_template(is_query=False, text=t, image_path=i, add_generation_prompt=False, description=d) \
                                            for t,i,d in zip(hard_negative_texts, hard_negative_images, hard_negative_description)]
                    if len(hard_negative_texts) != self.data_args.hard_negatives_per_sample:
                        raise ValueError(f"Insufficient hn ({len(hard_negative_texts)}) for {self.subset_name}")
            else:
                if self.model_backbone != PHI3V:
                    qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.model_backbone])
                    pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.model_backbone])

            query_texts.append(qry_text)
            pos_texts.append(pos_text)
            neg_texts.append(hard_negative_texts)
            # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images

            if self.data_args.fast_iter_with_no_visual:
                qry_image_path = pos_image_path = None
                
            qry_image = {"bytes": [None], "paths": [os.path.join(self.image_dir, qry_image_path) if qry_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(self.image_resolution, None)]}
            pos_image = {"bytes": [None], "paths": [os.path.join(self.image_dir, pos_image_path) if pos_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(self.image_resolution, None)]}
            neg_image = [{"bytes": [None], "paths": [os.path.join(self.image_dir, image_path) if image_path else None], "resolutions": [RESOLUTION_MAPPING.get(self.image_resolution, None)]} for image_path in hard_negative_images]
            query_images.append(qry_image)
            pos_images.append(pos_image)
            neg_images.append(neg_image)

            query_ecrs.append(query_description)
            pos_ecrs.append(pos_description)
        if len(query_texts) == 0:
            print('something went wrong')
        # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
        return {"query_text": query_texts, "query_image": query_images,
                "pos_text": pos_texts, "pos_image": pos_images,
                "neg_text": neg_texts, "neg_image": neg_images,
                "query_ecr": query_ecrs, "pos_ecr": pos_ecrs}

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
        query_ecrs, pos_ecrs = [], []
        batch_size = len(next(iter(batch_dict.values())))

        for data_idx in range(batch_size):
            one_sample = self._process_one_sample(data_idx, batch_dict, *args, **kwargs)
            query_text, query_image, pos_text, pos_image, neg_text, neg_image, query_description, target_description, neg_description  = \
                one_sample['query_text'], one_sample['query_image'], \
                one_sample['pos_text'], one_sample['pos_image'], \
                one_sample['neg_text'], one_sample['neg_image'], \
                one_sample['query_description'], one_sample['target_description'], \
                one_sample['neg_description']

            if (not query_image and self.data_args.rewrites_for_mm_only): # or self.encode_side == 'target':
                query_description = None
            if (not pos_image and self.data_args.rewrites_for_mm_only): # or self.encode_side == 'query':
                target_description = None
                neg_description = [None] * len(neg_description)


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
                
                query_text = self.format_text_for_chat_template(
                    is_query=True, text=query_text, image_path=query_image_input, video_path=query_video_input, 
                    description=query_description)
                pos_text = self.format_text_for_chat_template(
                    is_query=False, text=pos_text, image_path=pos_image_input, video_path=pos_video_input, 
                    description=target_description)
                
                if self.data_args.hard_negative_dir and neg_description:
                    formatted_neg_text = []
                    for text,image,description in zip(neg_text, neg_image, neg_description):
                        neg_image_input = neg_video_input = None
                        if image:
                            if len(image['paths']) > 1:
                                neg_video_input = image['paths'][0] or image['bytes'][0]
                            else:
                                neg_image_input = image['paths'][0] or image['bytes'][0]
                        formatted_neg_text.append(self.format_text_for_chat_template(
                            is_query=False, text=text, image_path=neg_image_input, video_path=neg_video_input, 
                            description=description))
                    neg_text = formatted_neg_text
                    if len(neg_text) != self.data_args.hard_negatives_per_sample:
                        raise ValueError(f"Insufficient hn ({len(neg_text)}) for {self.subset_name}")

            query_texts.append(query_text)
            query_images.append(query_image)
            pos_texts.append(pos_text)
            pos_images.append(pos_image)
            neg_texts.append(neg_text)
            neg_images.append(neg_image)

            query_ecrs.append(query_description)
            pos_ecrs.append(target_description)

        return {"query_text": query_texts, "query_image": query_images,
                "pos_text": pos_texts, "pos_image": pos_images,
                "neg_text": neg_texts, "neg_image": neg_images,
                "query_ecr": query_ecrs, "pos_ecr": pos_ecrs}


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