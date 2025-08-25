from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from datasets import Features, Value, Sequence, IterableDataset
import pickle, os
from datasets import load_dataset, concatenate_datasets
from ...model.processor import PHI3V, VLM_IMAGE_TOKENS, VLM_VIDEO_TOKENS
from ...utils import print_master, print_rank
from ..prompts import (format_description,
                       extract_query, extract_target,
                       IMAGE_TASKS,
                       TASK2ID)
from ..utils.dataset_utils import sample_dataset
from ..utils.vision_utils import save_frames, load_frames, sample_frames
from ...model.processor import process_input_text
import torch
from ..loader.mixed_dataset import add_metainfo_hook

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
        self.subset_name = self.dataset_config.get("subset_name", None) or self.dataset_name
        self.dataset_split = self.dataset_config.get("dataset_split", "original")
        self.image_dir = self.dataset_config.get('image_dir')
        self.model_backbone = self.model_args.model_backbone
        self.image_resolution = self.dataset_config.get('image_resolution')
        self.instruction = instruction

        self.query_key_text = query_key_text
        self.query_key_mm = query_key_mm
        self.cand_key_text = cand_key_text
        self.cand_key_mm = cand_key_mm

        self.query_descriptions = self.target_descriptions = None
        if data_args.query_description_dir is not None:
            desc_path = os.path.join(data_args.query_description_dir, self.dataset_name, "cot", "query.pkl")
            if os.path.exists(desc_path):
                with open(desc_path, "rb") as f:
                    self.query_descriptions = pickle.load(f)

        if data_args.target_description_dir is not None:
            desc_path = os.path.join(data_args.target_description_dir, self.dataset_name, "cot", "target.pkl")
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
            self.dataset = self.dataset.to_iterable_dataset(num_shards=n_workers_per_node)
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

        if self.model_args.do_sft_target and not self.model_args.do_cl and self.target_descriptions is not None:
            target_dataset = self.dataset.map(lambda row: {
                "query_text": row["pos_text"],
                "query_image": row["pos_image"],
                "pos_text": row["pos_text"],
                "pos_image": row["pos_image"], 
                "neg_text": row["neg_text"],
                "neg_image": row['neg_image'],
                "global_dataset_name": row["global_dataset_name"],
                "task_id": row["task_id"]
            }, batched=False)
            target_dataset = target_dataset.cast(MULTIMODAL_FEATURES)
            self.dataset = concatenate_datasets([self.dataset, target_dataset])

            num_rows *= 2
            
        print_master(f"Loaded {self.data_parser_name}/{self.dataset_name} dataset with {num_rows} samples")
        
        if not self.data_args.debug_prompt and isinstance(self.dataset, IterableDataset):
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
            instruction = self.instruction['query'] if self.instruction is not None else None
        else:
            extract_fn = extract_target
            instruction = self.instruction['target'] if self.instruction is not None else None

        text = extract_fn(text, self.subset_name)
        # make sure no extra visual tokens are left in the text
        text = text.replace(VLM_IMAGE_TOKENS[self.model_backbone], "")
        text = text.replace(VLM_VIDEO_TOKENS[self.model_backbone], "").strip()

        if instruction is not None:
            text = instruction.format(text=text)

        description = format_description(description, self.data_args.prompt_format)

        formatted_sample = []
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
                print("empty inputs")
                continue

            if data_args.apply_chat_template:
                query_description = pos_description = None
                if self.query_descriptions:
                    query_description = self.query_descriptions[(qry_text, qry_image_path)]
                if self.target_descriptions:
                    pos_description = self.target_descriptions[(pos_text, pos_image_path)]
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
                one_sample['neg_text'], one_sample['neg_image']

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

            query_texts.append(query_text)
            query_images.append(query_image)
            pos_texts.append(pos_text)
            pos_images.append(pos_image)
            neg_texts.append(neg_text)
            neg_images.append(neg_image)

        return {"query_text": query_texts, "query_image": query_images,
                "pos_text": pos_texts, "pos_image": pos_images,
                "neg_text": neg_texts, "neg_image": neg_images,}
