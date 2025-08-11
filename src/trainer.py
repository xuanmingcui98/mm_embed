import collections
import contextlib
import functools
import shutil
import sys
import time
from datetime import timedelta
from collections import defaultdict
from packaging import version
from accelerate import skip_first_batches, DistributedType, InitProcessGroupKwargs
from transformers import PretrainedConfig
from transformers.trainer import Trainer, TRAINING_ARGS_NAME, TRAINER_STATE_NAME
import torch.distributed as dist
from typing import Optional
import os
import torch
import math
from typing import Dict, Any, Union, List, Tuple

from torch import nn

from src.data.collator.train_collator import split_vlm_inputs, get_dense_rep, split_and_process_vlm_inputs
from src.model.model import MMEBModel
from src.loss import SimpleContrastiveLoss, DistributedContrastiveLoss
from src.grad_cache.grad_cache import GradCache
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.trainer_callback import (
    ExportableState,
    TrainerState,
)
from transformers.trainer_utils import (
    TrainOutput,
    has_length,
    speed_metrics, seed_worker,
)

from transformers.trainer_pt_utils import (
    get_model_param_count,
)

from transformers.trainer import FSDP_MODEL_NAME
from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_accelerate_available,
    is_apex_available,
    is_torch_xla_available,
    logging, is_sagemaker_mp_enabled,
    CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME
)

from src.utils import batch_to_device
from src.utils import print_master, print_rank

if is_apex_available():
    from apex import amp

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        pass
else:
    IS_XLA_FSDPV2_POST_2_2 = False

logger = logging.get_logger(__name__)

class MMEBTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.data_args = kwargs.pop("data_args", None)
        self.training_args = kwargs.get("args")
        self.model_args = kwargs.get("model").model_config

        super(MMEBTrainer, self).__init__(*args, **kwargs)

        self.is_ddp = dist.is_initialized()
        self.processor = self.processing_class
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self._signature_columns = ['query_text', 'query_image', 'target_text', 'target_image', 'label_ids', 'label']

    def get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break
        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.item()
        return batch_samples, num_items_in_batch

    def compute_loss(self, model, inputs, *args, **kwargs):
        qry_inputs, tgt_inputs = inputs
        return model(qry=qry_inputs, tgt=tgt_inputs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        if self.model.pooling_module is not None:
            pooling_module_state_dict = self.model.pooling_module.state_dict()
            pooling_module_state_dict = {"pooling_module." + k if not k.startswith("pooling_module.") else k : v for k, v in pooling_module_state_dict.items()}
            self.model.pooling_module.save_pretrained(
                os.path.join(output_dir, "pooling_module"), state_dict=pooling_module_state_dict, safe_serialization=False)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


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

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.model_args.checkpoint_path = resume_from_checkpoint
        print_master(f"Loading checkpoint from {resume_from_checkpoint}")
        self.model = MMEBModel.load(self.model_args, self.data_args)
        self.model_wrapped = self.model

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        print_master("***** Running training *****")
        print_master(f"  Num examples = {num_examples:,}")
        print_master(f"  Num Epochs = {num_train_epochs:,}")
        print_master(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            print_master(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        print_master(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        print_master(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print_master(f"  Total optimization steps = {max_steps:,}")
        print_master(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # @ruimeng use steps_trained_in_current_epoch to skip batches for finding buggy data
        # steps_trained_in_current_epoch = 42

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            print_master("  Continuing training from checkpoint, will skip to saved global_step")
            print_master(f"  Continuing training from epoch {epochs_trained}")
            print_master(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                print_master(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = defaultdict(lambda: torch.tensor(0.0, device=args.device))
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = defaultdict(float)
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader.dataset, "set_epoch"):
                # print(f'\t\tSetting new epoch={epoch}')
                epoch_dataloader.dataset.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            num_items_in_batch = None
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    total_batched_samples += 1

                    dataset_stat = collections.Counter(inputs[0]['global_dataset_name'])
                    # print_rank(f"dataset name: {str(set(inputs[0]['global_dataset_name']))}")
                    # for dname, count in sorted(dataset_stat.items(), key=lambda t:t[1], reverse=True):
                    #     print_rank(f"\t\tdataset_name={dname}, count={count}")

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )
                    do_sync_step = is_last_step_and_steps_less_than_grad_acc or (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                    )
                    # Since we perform prefetching, we need to manually set sync_gradients
                    if not do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(False)
                    else:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).cpu().item()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if isinstance(tr_loss_step, torch.Tensor):
                        tr_loss_step = {"loss": tr_loss_step}
                    
                    for k,loss_step in tr_loss_step.items():
                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(loss_step) or torch.isinf(loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss[k] = tr_loss[k] + tr_loss[k] / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            if tr_loss[k].device != tr_loss_step[k].device:
                                raise ValueError(
                                    f"Calculated loss must be on the original device: {tr_loss[k].device} but device in use is {loss_step.device}"
                                )
                            tr_loss[k] = tr_loss[k] + tr_loss_step[k]

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = torch.nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, time.time())
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, time.time())

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        print_master("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        # self._total_loss_scalar += tr_loss.item()
        for k, v in tr_loss.items():
            self._total_loss_scalar[k] = self._total_loss_scalar.get(k, 0.0) + v.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        # train_loss = self._total_loss_scalar / effective_global_step
        for k, v in self._total_loss_scalar.items():
            self._total_loss_scalar[k] = v / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        # metrics["train_loss"] = train_loss
        for k, v in self._total_loss_scalar.items():
            metrics[f"train_{k}"] = v

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    print_master(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, self._total_loss_scalar['loss'], metrics)

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss_dict = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            loss = loss_dict['loss']

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            for k,v in loss_dict.items():
                loss_dict[k] = v.mean().detach()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

                for k,v in loss_dict.items():
                    loss_dict[k] = v / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss_dict
        

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            for k,loss in tr_loss.items():
                loss = self._nested_gather(loss).mean().item()

            # reset tr_loss to zero
            # tr_loss -= tr_loss
            
                logs[k] = round(loss / (self.state.global_step - self._globalstep_last_logged), 4)
                self._total_loss_scalar[k] += loss

                tr_loss[k] -= tr_loss[k]
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                learning_rates = self.lr_scheduler.get_last_lr()
                # if len(learning_rates) > 1:
                #     logs['llm_learning_rate'] = learning_rates[0]
                #     logs['pooler_learning_rate'] = learning_rates[1]
                # else:
                #     if self.args.freeze_base_model:
                #         logs["pooler_learning_rate"] = learning_rates[0]
                #     else:
                #         logs["llm_learning_rate"] = learning_rates[0]
                if self.model.pooling_module is None:
                    logs["llm_learning_rate"] = learning_rates[0]
                else:
                    if self.model.model_config.freeze_base_model:
                        logs["pooler_learning_rate"] = learning_rates[0]
                    else:
                        logs["llm_learning_rate"] = learning_rates[0]
                        logs["pooler_learning_rate"] = learning_rates[1]

            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            
class GradCacheLateProcessTrainer(MMEBTrainer):
    """
    Adapted from gradcache repo.
    """
    def __init__(self, *args, **kwargs):
        super(GradCacheLateProcessTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        loss_fn_cls = DistributedContrastiveLoss if self.is_ddp else SimpleContrastiveLoss
        loss_fn = loss_fn_cls(temperature=self.model.temperature, inter_task_temperature=self.args.inter_task_temperature, use_symmetric_loss=self.args.use_symmetric_loss)
        # process_fn = functools.partial(process_vlm_inputs_fns[self.args.model_backbone], processor=self.processing_class, max_length=self.max_length)

        self.gc = GradCache(
            training_args=self.args,
            data_args=self.data_args,
            model_args=self.model_args,
            accelerator=self.accelerator,
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_and_process_vlm_inputs,
            # process_fn=process_fn,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        model.train()
        queries, targets = inputs
        # queries = batch_to_device(queries, model.device)
        # targets = batch_to_device(targets, model.device)
        device = next(model.parameters()).device
        queries = batch_to_device(queries, device)
        targets = batch_to_device(targets, device)
        queries, targets = {'qry': queries}, {'tgt': targets}

        _distributed = self.args.local_rank > -1
        if _distributed:
            self.gc.models = [model, model]
            loss_dict = self.gc(queries, targets, no_sync_except_last=_distributed)
        else:
            loss_dict = model(queries, targets)

        if 'cl_loss' in loss_dict:
            loss_dict['cl_loss'] = loss_dict['cl_loss'] / self._dist_loss_scale_factor
        return loss_dict


    # def _save(self, output_dir: Optional[str] = None, state_dict=None):
    #     print_master(f"Saving model to {output_dir}")
    #     os.makedirs(output_dir, exist_ok=True)

    #     if state_dict is None:
    #         state_dict = self.model.state_dict()
    #     prefix = 'encoder.'
    #     assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
    #     state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    #     self.model.encoder.save_pretrained(
    #         output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
    #     )

    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)

    #     torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    #     self.model.encoder.config.to_json_file(os.path.join(output_dir, 'config.json'))
