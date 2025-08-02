from torch.utils.data import DataLoader
from typing import Optional
import torch
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import (
    TrainOutput,
    has_length,
    speed_metrics, seed_worker,
)

class MixedInputSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    def get_batch_samples(self, epoch_iterator, num_batches, device=None):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break
        # if len(batch_samples) > 0 and "labels" in batch_samples[0]:
        #     # For now we don't support object detection
        #     try:
        #         num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
        #     except (TypeError, AttributeError):
        #         pass
        # if self.args.average_tokens_across_devices and num_items_in_batch is not None:
        #     num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()
        # if torch.is_tensor(num_items_in_batch):
        #     num_items_in_batch = num_items_in_batch.item()
        return batch_samples, None

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # override original trainer's method
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        override original trainer's method to disable self.accelerator.prepare since it will wrap DataLoaderDispatcher and lead to
        (1) `RuntimeError: You can't use batches of different size with `dispatch_batches=True` or when using an `IterableDataset`.`
        (2) all outputs of dataloader must be tensors
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        else:
            dataloader_params["sampler"] = None
            dataloader_params["shuffle"] = False
            dataloader_params["drop_last"] = True
            dataloader_params["prefetch_factor"] = None # # tune on both prefetch_factor and persistent_workers will cause hang at epoch2
        return DataLoader(train_dataset, **dataloader_params)