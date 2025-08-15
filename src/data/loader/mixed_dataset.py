from abc import ABCMeta
from collections import defaultdict
from typing import Union, List, Dict, Any
from functools import wraps

from datasets.distributed import split_dataset_by_node
from datasets import concatenate_datasets
from ..dataset.hf_datasets import interleave_datasets
from ..prompts import IMAGE_TASKS, TASK_TYPE
from src.utils import print_master
import torch

class AutoPairDataset(metaclass=ABCMeta):
    # Base class for auto datasets.
    registry = {}
    instruction_registry = defaultdict(None)

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
            if kwargs.get("dataset_name") is not None:
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
                print(f"[Alert] AutoPairDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def register_instruction(cls, dataset_name: Union[str, List[str]], instruction):
        """
        Register the instruction for the dataset.
        """
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]
        for name in dataset_name:
            if name in cls.instruction_registry:
                print(f"[Alert] AutoPairDataset: a instruction in the same name ({name}) has been registered")
            else:
                cls.instruction_registry[name] = instruction

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
        return batch_data

    return wrapper


def init_mixed_dataset(dataset_config, model_args, data_args, training_args, processor):

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()

    weights = [d['weight'] for d in dataset_config.values()]
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

    if len(train_datasets) > 1:
        train_dataset = interleave_datasets(train_datasets, probabilities=probs, batch_size=interleave_batch_size,
                                            seed=training_args.seed, stopping_strategy=training_args.interleave_stopping_strategy)
    else:
        train_dataset = train_datasets[0]
    
    if torch.distributed.is_initialized():
        train_dataset = split_dataset_by_node(train_dataset, rank=torch.distributed.get_rank(), world_size=world_size)

    if not data_args.debug_prompt:
        setattr(train_dataset, "num_rows", total_num_rows)

    return train_dataset

def init_sft_dataset(dataset_config, model_args, data_args, training_args, processor):
    """
    Initialize the dataset for supervised fine-tuning (SFT).
    We don't use interleave datasets/weighted datasets for SFT.
    """

    train_datasets = []
    for data_idx, (global_dataset_name, dataset_config) in enumerate(dataset_config.items()):
        train_dataset = AutoPairDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, processor=processor, **dataset_config)
        print_master(f"\t\tDataset#{data_idx} (dataset_parser={dataset_config.get('dataset_parser', 'n/a')}): {global_dataset_name}, num_rows={train_dataset.num_rows}")
        train_datasets.append(train_dataset)

    total_num_rows = sum([d.num_rows for d in train_datasets])
    from datasets import concatenate_datasets
    train_datasets = concatenate_datasets(train_datasets)
    setattr(train_datasets, "num_rows", total_num_rows)
    return train_datasets



