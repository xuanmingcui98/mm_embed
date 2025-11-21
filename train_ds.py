# Adapted from Tevatron code
import logging
import os.path
import sys
import os

os.environ['TZ'] = "America/New_York"

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)

import torch
import wandb
import yaml
import tqdm  
import json
from functools import wraps
from transformers import HfArgumentParser, set_seed
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.train_collator import MultimodalDataCollator, ContrastiveDataCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.model import MMEBModel
from src.trainer import GradCacheLateProcessTrainer, MMEBTrainer
from src.utils import print_rank, print_master, find_latest_checkpoint

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

logging.getLogger("PIL").setLevel(logging.WARNING)
    
original_tqdm_init = tqdm.__init__

@wraps(original_tqdm_init)
def new_tqdm_init(self, *args, **kwargs):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() > 0:
        kwargs['disable'] = True
        
    return original_tqdm_init(self, *args, **kwargs)

tqdm.__init__ = new_tqdm_init


def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    set_seed(training_args.seed)

    training_args.ddp_timeout = 4800  # Set a timeout for DDP to avoid hanging

    # Check for existing checkpoints
    if training_args.resume_from == 'auto':
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    else:
        resume_checkpoint_dir = None
        logger.info("No checkpoint found. Starting fresh training.")

    model, processor = MMEBModel.build(model_args, data_args)

    setattr(training_args, 'model_backbone', model_args.model_backbone)

    with training_args.main_process_first(local=False):
        with open(data_args.dataset_config, 'r') as yaml_file:
            dataset_config = yaml.safe_load(yaml_file)
            train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args, processor)

    if 'qwen2_5_vl' in model_args.model_backbone or 'qwen3_vl' in model_args.model_backbone:
        train_collator = ContrastiveDataCollator(processor)
    else:
        train_collator = MultimodalDataCollator(processor, model_args, data_args, training_args, batch_size=training_args.per_device_train_batch_size)

    trainer_cls = GradCacheLateProcessTrainer if training_args.grad_cache else MMEBTrainer

    if model_args.do_cl:
        # force it to be 1
        training_args.gradient_accumulation_steps = 1
        
    print_master(f"World size: {training_args.world_size}")
    training_args.max_steps = training_args.max_steps \
          if training_args.max_steps > 0 else \
          training_args.num_train_epochs * train_dataset.num_rows // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)

    # if "32b" in model_args.model_name.lower():
    setattr(training_args, "ddp_find_unused_parameters", False)
    setattr(training_args, "gradient_checkpointing_kwargs", {"use_reentrant": True})
    # do_gradient_checkpointing = not (training_args.deepspeed and "zero3" in training_args.deepspeed)
    setattr(training_args, "gradient_checkpointing", True)

    trainer = trainer_cls(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        data_args=data_args,
    )

    if model.pooling_module is not None and training_args.pooler_learning_rate is not None:
        if model_args.freeze_base_model:
            optimizer_grouped_parameters = [
                {"params": model.pooling_module.parameters(), "lr": training_args.pooler_learning_rate},
                ]
        else:
            optimizer_grouped_parameters = [
                {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": training_args.learning_rate},
                {"params": model.pooling_module.parameters(), "lr": training_args.pooler_learning_rate},
                ]
            
        optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(trainer.args)
        trainer.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    train_dataset.trainer = trainer

    # Initialize WandB if enabled
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)

    print_master(f'model_args: {json.dumps(vars(model_args), indent=2)}')
    print_master(f'data_args: {json.dumps(vars(data_args), indent=2)}')


    def dump():
        os.makedirs(training_args.output_dir, exist_ok=True)
        json.dump(vars(model_args), open(os.path.join(training_args.output_dir, "model_args.json"), 'w'), indent=2)
        json.dump(vars(data_args), open(os.path.join(training_args.output_dir, "data_args.json"), 'w'), indent=2)

        training_args_to_save = {
            'output_dir': training_args.output_dir,
            'num_train_epochs': training_args.num_train_epochs,
            'per_device_train_batch_size': training_args.per_device_train_batch_size,
            'grad_cache': training_args.grad_cache,
            'gc_q_chunk_size': training_args.gc_q_chunk_size,
            'gc_p_chunk_size': training_args.gc_p_chunk_size,
            'pooler_learning_rate': training_args.pooler_learning_rate,
            'cl_loss_scalar': training_args.cl_loss_scalar,
            'learning_rate': training_args.learning_rate,
            'warmup_ratio': training_args.warmup_ratio,
            'weight_decay': training_args.weight_decay,
        }
        json.dump(training_args_to_save, open(os.path.join(training_args.output_dir, "training_args.json"), 'w'), indent=2)

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            dump()
    else:
        dump()

    trainer.train(resume_from_checkpoint=resume_checkpoint_dir)
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
