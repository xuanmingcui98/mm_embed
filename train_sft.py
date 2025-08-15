# Adapted from Tevatron code
import logging
import sys
import torch
import wandb
from functools import wraps

from transformers import (
    HfArgumentParser,
    set_seed
)
import json
import os

os.environ['TZ'] = "America/Los_Angeles"

from src.dataset import TrainTextImageDataset
from src.collator import TrainTextImageDataCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.model_utils import load_processor, get_backbone_name
from src.trainer import GradCacheLateProcessTrainer, MMEBTrainer
from src.utils import print_rank, print_master

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

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

    training_args.freeze_base_model = model_args.freeze_base_model
    training_args.do_sft = model_args.do_sft
    training_args.do_cl = model_args.do_cl

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
    
    model, processor = MMEBModel.build(model_args)
    model_backbone = get_backbone_name(hf_config=model.config)

    setattr(training_args, 'model_backbone', model_backbone)
    print_master(f'model_backbone: {model_backbone}')

    # model_args.model_backbone = "qwen2_vl"

    train_dataset = TrainTextImageDataset(data_args, model_args)
    collator = TrainTextImageDataCollator(data_args, model_args, processor)


    trainer_cls = GradCacheLateProcessTrainer if training_args.grad_cache else MMEBTrainer

    trainer = trainer_cls(
        model=model,
        training_args=training_args,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
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

    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online", config={**vars(model_args), **vars(data_args)})

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
