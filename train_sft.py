# Adapted from Tevatron code
import logging
import os.path
import sys
import os

os.environ['TZ'] = "America/Los_Angeles"

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
from transformers import HfArgumentParser, set_seed, AutoConfig
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.train_collator import MultimodalDataSFTCollator
from src.data.loader.mixed_dataset import init_sft_dataset
from src.model.model import MMEBModel
from src.trainer import GradCacheLateProcessTrainer, MMEBTrainer
from src.utils import print_rank, print_master, find_latest_checkpoint
from datasets.utils.logging import disable_progress_bar
from src.model.processor import load_processor, get_visual_token_ids
from trl import SFTTrainer, SFTConfig
from src.trainer_sft import MixedInputSFTTrainer
from src.model.processor import get_backbone_name
from accelerate import PartialState



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

    # Check for existing checkpoints
    if training_args.resume_from == 'auto':
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            print_master(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            print_master(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    else:
        resume_checkpoint_dir = None
        print_master("No checkpoint found. Starting fresh training.")


    if 'qwen2.5' in model_args.model_name.lower():
        from src.model.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration

    elif 'qwen2' in model_args.model_name.lower():
        from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
        model_cls = Qwen2VLForConditionalGeneration
    else:
        raise NotImplementedError(f"Model {model_args.model_name} not implemented")

    config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)

    model_backbone = get_backbone_name(hf_config=config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    config.padding_side = "left"
    config.use_cache = False

    device_string = PartialState().process_index
    model = model_cls.from_pretrained(
        model_args.model_name, 
        config=config,
        attn_implementation="flash_attention_2",
        # device_map=device_string,
        torch_dtype=torch.bfloat16)
    processor = load_processor(model_args, data_args=data_args)

    lora_config = None
    if model_args.lora:
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(','),
            lora_dropout=model_args.lora_dropout,
            task_type="CAUSAL_LM",
            # init_lora_weights="gaussian",
            use_dora=True,
            inference_mode=False
        )

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)
    
    train_dataset = init_sft_dataset(dataset_config, model_args, data_args, training_args, processor)
    train_collator = MultimodalDataSFTCollator(processor, model_args, data_args, training_args)

    print(f"Before trainer initialization. World size: {training_args.world_size}")

    sft_config = SFTConfig(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.num_train_epochs * train_dataset.num_rows // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size),
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        optim="adamw_torch_fused",
        save_strategy="steps",
        save_steps=training_args.save_steps,
        bf16=training_args.bf16,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_ratio=training_args.warmup_ratio,
        lr_scheduler_type=training_args.lr_scheduler_type,
        dataset_kwargs={"skip_prepare_dataset" : True},
        # use_liger_kernel=True,
        report_to="none",
        remove_unused_columns=False,
        logging_steps=training_args.logging_steps)

    trainer = MixedInputSFTTrainer(
        model=model,
        processing_class=processor,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=train_collator,
        peft_config=lora_config,
    )

    print(f"After trainer initialization. World size: {training_args.world_size}")

    print_master(f'model_args: {json.dumps(vars(model_args), indent=2)}')
    print_master(f'data_args: {json.dumps(vars(data_args), indent=2)}')

    if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):

        os.makedirs(training_args.output_dir, exist_ok=True)
        json.dump(vars(model_args), open(os.path.join(training_args.output_dir, "model_args.json"), 'w'), indent=2)
        json.dump(vars(data_args), open(os.path.join(training_args.output_dir, "data_args.json"), 'w'), indent=2)

        training_args_to_save = {
            'output_dir': training_args.output_dir,
            'num_train_epochs': training_args.num_train_epochs,
            'per_device_train_batch_size': training_args.per_device_train_batch_size,
            'learning_rate': training_args.learning_rate,
            'warmup_ratio': training_args.warmup_ratio,
            'weight_decay': training_args.weight_decay,
            'lora': model_args.lora,
            'lora_r': model_args.lora_r,
            'lora_alpha': model_args.lora_alpha,
            'lora_dropout': model_args.lora_dropout,
        }
        json.dump(training_args_to_save, open(os.path.join(training_args.output_dir, "training_args.json"), 'w'), indent=2)
        print_master(f'Training arguments: {json.dumps(training_args_to_save, indent=2)}')

        # if 'wandb' in training_args.report_to:
        #     print_master('init wandb')
        #     wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")
        #     wandb.config.update(model_args)
        #     wandb.config.update(data_args)
        #     wandb.config.update(training_args)

    trainer.train(resume_from_checkpoint=resume_checkpoint_dir)
    trainer.save_model()

if __name__ == "__main__":
    main()
