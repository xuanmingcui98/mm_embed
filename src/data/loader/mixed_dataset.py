from abc import ABCMeta
from collections import defaultdict
from typing import Union, List, Dict, Any
from functools import wraps
import yaml, os
import datasets
from datasets import load_from_disk
from datasets.distributed import split_dataset_by_node
from datasets import concatenate_datasets
from ..dataset.hf_datasets import interleave_datasets
from ..prompts import TASK2ID
from src.utils import print_master
import torch

class AutoPairDataset(metaclass=ABCMeta):
    # Base class for auto datasets.
    registry = {}
    instruction_registry = defaultdict(lambda: None)

    def __init_subclass__(cls):
        if cls.__name__ not in cls.registry:
            cls.registry[cls.__name__] = cls
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
            if kwargs.get("subset_name") is not None:
                instruction = cls.instruction_registry[kwargs["subset_name"]]
            elif kwargs.get("dataset_name") is not None:
                instruction = cls.instruction_registry[kwargs["dataset_name"]]
            else:
                instruction = cls.instruction_registry[dataset_parser]
            return cls.registry[dataset_parser](*args, **kwargs, instruction=instruction).load()
        except Exception as e:
            raise e

    @classmethod
    def register(cls, dataset_name):
        def inner_wrapper(wrapped_class):
            if dataset_name in cls.registry:
                print(f"[Alert] AutoPairEvalDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def register_instruction(cls, dataset_name: Union[str, List[str]], instruction):
        """
        Register the instruction for the dataset.
        """
        def inner_wrapper(wrapped_class):
            if isinstance(dataset_name, str):
                dataset_names = [dataset_name]
            else:
                dataset_names = dataset_name
            for name in dataset_names:
                if name in cls.instruction_registry:
                    print(f"[Alert] AutoPairEvalDataset: a instruction in the same name ({name}) has been registered")
                else:
                    cls.instruction_registry[name] = instruction
            return wrapped_class
        return inner_wrapper

class AutoSFTDataset(metaclass=ABCMeta):
    # Base class for auto datasets.
    registry = {}

    def __init_subclass__(cls):
        if cls.__name__ not in AutoSFTDataset.registry:
            AutoSFTDataset.registry[cls.__name__] = cls
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
                print(f"[Alert] AutoSFTDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
# create a copy for the AutoPairEvalDataset class
class AutoPairEvalDataset(AutoPairDataset):
    registry = {}
    instruction_registry = defaultdict(lambda: None)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def add_metainfo_hook(f):
    """
    A post-processing wrapper function that add meta information (e.g. data_type, dataset_name, loss_type) into batches
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # go through data pipeline customized to each dataset
        batch_data = f(*args, **kwargs)
        # append common metadata
        batch_size = len(batch_data.get('query_text', batch_data.get('cand_text', [])))
        global_dataset_name = kwargs.get("global_dataset_name", "None")
        batch_data['global_dataset_name'] = [global_dataset_name] * batch_size
        batch_data['task_id'] = [TASK2ID[kwargs.get("subset_name", kwargs.get("dataset_name"))]] * batch_size
        return batch_data

    return wrapper


def init_mixed_dataset(dataset_config, model_args, data_args, training_args, processor, is_iterable=True):

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()

    weights = [d.get('weight', 1) for d in dataset_config.values()]
    w_sum = sum(weights)
    probs = [w / w_sum for w in weights]
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    train_datasets = []

    for data_idx, (global_dataset_name, dataset_config) in enumerate(dataset_config.items()):
        dataset_config['world_size'] = world_size
        train_dataset = AutoPairDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, processor=processor, **dataset_config)
        print_master(f"\t\tDataset#{data_idx} (dataset_parser={dataset_config.get('dataset_parser', 'n/a')}): {global_dataset_name}, num_rows={train_dataset.num_rows}, prob={probs[data_idx] * 100.0}")
        train_datasets.append(train_dataset)

    if training_args.interleave_batch_size and training_args.interleave_batch_size <= 1.0:
        interleave_batch_size = training_args.per_device_train_batch_size * world_size * training_args.interleave_batch_size
    else:
        interleave_batch_size = training_args.interleave_batch_size
    total_num_rows = sum([d.num_rows for d in train_datasets])
    print_master(f"\nInitializing interleave datasets:"
                 f"\n\t\tworld_size={world_size}"
                 f"\n\t\ttotal num rows={total_num_rows}"
                 f"\n\t\tglobal batch size={training_args.per_device_train_batch_size * world_size}"
                 f"\n\t\testimated num step per epoch={total_num_rows/(training_args.per_device_train_batch_size * world_size)}"
                 f"\n\t\tinterleave_batch_size={interleave_batch_size}"
                 )
    assert total_num_rows >= (training_args.per_device_train_batch_size * world_size), \
        f"total_num_rows(={total_num_rows}) must be greater than or equal to global batch size (={training_args.per_device_train_batch_size * world_size}), since the last batch will be dropped."

    if is_iterable:
        if len(train_datasets) > 1:
            train_dataset = interleave_datasets(train_datasets, probabilities=probs, batch_size=interleave_batch_size,
                                                seed=training_args.seed, stopping_strategy=training_args.interleave_stopping_strategy)
        else:
            train_dataset = train_datasets[0]
        
        if torch.distributed.is_initialized():
            train_dataset = split_dataset_by_node(train_dataset, rank=torch.distributed.get_rank(), world_size=world_size)

        if not data_args.debug_prompt:
            setattr(train_dataset, "num_rows", total_num_rows)
    else:
        train_dataset = datasets.concatenate_datasets(train_datasets)

    return train_dataset

def init_sft_dataset(dataset_config, model_args, data_args, training_args, processor):
    """
    Initialize the dataset for supervised fine-tuning (SFT).
    We don't use interleave datasets/weighted datasets for SFT.
    """

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)

    train_datasets = []
    for data_idx, (global_dataset_name, dataset_config) in enumerate(dataset_config.items()):
        cached_dataset_path = os.path.join(data_args.cache_dataset_dir, dataset_config.get("subset_name", dataset_config.get("dataset_name")))
        if os.path.exists(cached_dataset_path) and (not data_args.rebuild_cache):
            print_master(f"Found cached dataset for {global_dataset_name} at {os.path.join(data_args.cache_dataset_dir, dataset_config.get('subset_name', dataset_config.get('dataset_name')))}, loading it ...")
            train_dataset = load_from_disk(cached_dataset_path)
        else:
            train_dataset = AutoSFTDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, processor=processor, **dataset_config)
            if data_args.cache_dataset_dir:
                train_dataset.save_to_disk(cached_dataset_path)

        num_sample_per_subset = dataset_config.get("num_sample_per_subset", 1e9)
        if train_dataset.num_rows >= num_sample_per_subset:
            train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(num_sample_per_subset))
        print_master(f"\t\tDataset#{data_idx} (dataset_parser={dataset_config.get('dataset_parser', 'n/a')}): {global_dataset_name}, num_rows={train_dataset.num_rows}")
        train_datasets.append(train_dataset)

    from datasets import concatenate_datasets
    train_datasets = concatenate_datasets(train_datasets)
    train_datasets = train_datasets.shuffle(seed=training_args.seed)
    return train_datasets


