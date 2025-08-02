from abc import ABCMeta, abstractmethod
from functools import wraps
from datasets import Features, Value, Sequence
import pickle, os
from datasets import load_dataset, concatenate_datasets
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.utils import print_master, print_rank
from torch.utils.data import Dataset
from ..prompts import (get_query, get_target, 
                       format_description, format_text_for_chat_template, 
                       TASK2ID)
from functools import partial
import torch


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


class AutoPairDataset(metaclass=ABCMeta):
    # Base class for auto datasets.
    registry = {}

    def __init_subclass__(cls):
        if cls.__name__ not in AutoPairDataset.registry:
            AutoPairDataset.registry[cls.__name__] = cls
        else:
            raise RuntimeError('Subclass "{cls.__name__}" has already defined.')

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def instantiate(cls, dataset_parser, *args, **kwargs):
        try:
            return cls.registry[dataset_parser](*args, **kwargs).load()
        except Exception as e:
            raise e

    @classmethod
    def register(cls, dataset_name):
        def inner_wrapper(wrapped_class):
            if dataset_name in cls.registry:
                print(f"[Alert] AutoPairDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @abstractmethod
    def main(self):
        pass

def add_metainfo_hook(f):
    """
    A post-processing wrapper function that add meta information (e.g. data_type, dataset_name, loss_type) into batches
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # go through data pipeline customized to each dataset
        batch_data = f(*args, **kwargs)
        # append common metadata
        batch_size = len(batch_data['query_text'])
        global_dataset_name = kwargs.get("global_dataset_name", "None")
        batch_data['global_dataset_name'] = [global_dataset_name] * batch_size
        batch_data['task_id'] = [TASK2ID[kwargs['subset_name']]] * batch_size
        return batch_data

    return wrapper

class BaseDatasetProcessor:
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

        self.dataset_name = self.dataset_config.get("dataset_name", self.data_parser_name)
        self.subset_name = self.dataset_config.get("subset_name")
        self.dataset_split = self.dataset_config.get("dataset_split", "original")

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

    def load(self):
        dataset = self._load_hf_dataset()

        num_rows = dataset.num_rows

        self.dataset_config['query_descriptions'] = self.query_descriptions
        self.dataset_config['target_descriptions'] = self.target_descriptions
        self.dataset_config['model_backbone'] = self.model_args.model_backbone
        self.dataset_config['image_resolution'] = self.data_args.image_resolution
        self.dataset_config['global_dataset_name'] = f'{self.data_parser_name}/{self.subset_name}'

        remove_columns = ['qry', 'qry_image_path', 'pos_image_path']
        if 'neg_image_path' in self.column_names:
            remove_columns.append('neg_image_path')

        process_fn = partial(self.batch_preprocess, data_args=self.data_args, model_args=self.model_args, processor=self.processor, **self.dataset_config)
        dataset = dataset.map(
                            lambda x:
                            # # format_fn(x), 
                            self.batch_preprocess(x, data_args=self.data_args, model_args=self.model_args, processor=self.processor, **self.dataset_config),
                            # process_fn,
                            batched=True, 
                            batch_size=2048,
                            remove_columns=remove_columns)

        dataset = dataset.cast(MULTIMODAL_FEATURES)

        if self.model_args.do_sft_target and not self.model_args.do_cl and self.target_descriptions is not None:
            target_dataset = dataset.map(lambda row: {
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
            dataset = concatenate_datasets([dataset, target_dataset])

            num_rows *= 2
            
        # dataset = dataset.add_column("global_dataset_name", [self.dataset_config.get("global_dataset_name", self.data_parser_name)] * num_rows)
        # dataset = dataset.add_column("task_id", [TASK2ID[self.subset_name]] * num_rows)

        print_master(f"Loaded {self.data_parser_name}/{self.subset_name} dataset with {num_rows} samples")
        setattr(dataset, 'num_rows', num_rows)
        
        return dataset

    def _load_hf_dataset(self):
        """
            Load the dataset based on the configuration.
            May be implemented by subclasses.
        """

        dataset = load_dataset(self.dataset_name, self.subset_name, split=f"{self.dataset_split}")

        self.column_names = dataset.column_names
        num_sample_per_subset = self.dataset_config.get("num_sample_per_subset", getattr(self.data_args, "num_sample_per_subset", None))
        if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
            num_rows = int(num_sample_per_subset)
            dataset = dataset.select(range(num_rows))
        num_rows = dataset.num_rows
        num_shards = self.training_args.dataloader_num_workers if self.training_args.dataloader_num_workers > 0 else 1
        dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards
        setattr(dataset, 'num_rows', num_rows)
        return dataset


    @add_metainfo_hook
    def batch_preprocess(self, batch_dict, data_args, model_args, processor, *args, **kwargs):
        image_dir = kwargs['image_dir']
        model_backbone = kwargs['model_backbone']
        image_resolution = kwargs['image_resolution']

        is_train = kwargs.get("dataset_split", "original") == "original"

        batch_size = len(batch_dict['qry'])
        query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
        for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path in \
            zip(batch_dict['qry'], batch_dict['qry_image_path'],
                batch_dict['pos_text'], batch_dict['pos_image_path'],
                batch_dict.get('neg_text', [''] * batch_size), batch_dict.get('neg_image_path', [None] * batch_size)):
            if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
                print("empty inputs")
                continue

            qry_description = pos_description = None
            if kwargs['query_descriptions'] is not None:
                qry_description = format_description(kwargs['query_descriptions'].get((qry_text, qry_image_path), None), data_args.use_cot)

            if kwargs['target_descriptions'] is not None:
                pos_description = format_description(kwargs['target_descriptions'].get((pos_text, pos_image_path), None), data_args.use_cot)

            if data_args.apply_chat_template:
                qry_text = get_query(kwargs['subset_name'], qry_text, data_args.use_cot)
                pos_text = get_target(kwargs['subset_name'], pos_text, data_args.use_cot)
                qry_text = format_text_for_chat_template(processor, qry_text, qry_image_path, description=qry_description, add_generation_prompt=not is_train and model_args.do_sft_query)
                pos_text = format_text_for_chat_template(processor, pos_text, pos_image_path, description=pos_description, add_generation_prompt=not is_train and model_args.do_sft_target)
            else:
                if model_backbone != PHI3V:
                    qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
                    pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
                    neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]) if neg_text else ''

            query_texts.append(qry_text)
            pos_texts.append(pos_text)
            neg_texts.append(neg_text)
            # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
            qry_image = {"bytes": [None], "paths": [os.path.join(image_dir, qry_image_path) if qry_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
            pos_image = {"bytes": [None], "paths": [os.path.join(image_dir, pos_image_path) if pos_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
            neg_image = {"bytes": [None], "paths": [os.path.join(image_dir, neg_image_path) if neg_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
            query_images.append(qry_image)
            pos_images.append(pos_image)
            neg_images.append(neg_image)
        if len(query_texts) == 0:
            print('something went wrong')
        # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
        return {"query_text": query_texts, "query_image": query_images,
                "pos_text": pos_texts, "pos_image": pos_images,
                "neg_text": neg_texts, "neg_image": neg_images}