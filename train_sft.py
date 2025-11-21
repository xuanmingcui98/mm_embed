
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO
from dataclasses import dataclass

import torch
from transformers import (
    set_seed
)
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
from peft import LoraConfig
from tqdm import tqdm
from PIL import Image
from transformers import HfArgumentParser, set_seed, AutoConfig
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.loader.mixed_dataset import init_sft_dataset

def is_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def tqdm0(it, **kw):
    kw.setdefault("disable", not is_main_process())
    return tqdm(it, **kw)

def safe_open_image(image):
    try:
        if isinstance(image, bytes):
            img = Image.open(BytesIO(image)).convert("RGB")
        elif isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise Exception
        return img
    except Exception as e:
        print(f"Error loading image {image}: {e}")
        return None

def train_transform(batch):
    images = []   # per-sample: [PIL.Image, ...] or []
    videos = []   # per-sample: [PIL.Image (frame), ...] or []
    messages = []

    for image, prompt, answer in zip(
        batch['image'], batch["prompt"], batch["answer"]
    ):
        # image is a dict of {"bytes": [...], "paths": [...]} 
        if not answer:
            continue

        # --- Load media ---
        sample_images = []
        sample_video_frames = []

        # Normalize to identify modality
        is_video = isinstance(image['paths'], (list, tuple)) and len(image['paths']) > 1

        if is_video:
            # Load frames; skip unreadable frames gracefully
            for p in image['paths']:
                frame = safe_open_image(p)
                sample_video_frames.append(frame)
        else:
            if image['bytes'][0] is not None:
                sample_images = [safe_open_image(image['bytes'][0])]
            elif image['paths'][0]:
                sample_images = [safe_open_image(image['paths'][0])]
            else:
                sample_images = []

        # --- Build messages ---
        user_content = []

        if sample_video_frames:
            user_content.append({"type": "video"})
        elif sample_images:
            user_content.append({"type": "image"})
        # Always include the text prompt
        user_content.append({"type": "text", "text": prompt})

        messages.append(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
        )

        images.append(sample_images)           # [] if none
        videos.append(sample_video_frames)     # [] if none

    return {"images": images, "videos": videos, "messages": messages}

def prepare_multimodal_messages(messages: list[dict[str, Any]], num_images: int, num_videos: int) -> None:

    for message in messages:
        if message["role"] == "system":
            if isinstance(message["content"], str):  # if already prepared, the content will be a list
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif message["role"] == "user":

            placeholders = [{"type": "image"}] * num_images + [{"type": "video"}] * num_videos
            message["content"] = [*placeholders, {"type": "text", "text": message["content"]}]

        elif message["role"] == "assistant":
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        else:
            raise ValueError(f"Invalid role in message: {message['role']}. Expected 'user', 'assistant', or 'system'.")

@dataclass
class MultimodalCollator(DataCollatorForVisionLanguageModeling):

    visual_token_ids: List[int] = (151652, 151653, 151654, 151655, 151656) # default to qwen-vl series

    def _collate_language_modeling(self, examples):
        images = [example["images"] for example in examples if example["images"]]
        videos = [example["videos"] for example in examples if example["videos"]]
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None

        if "messages" in examples[0]:  # conversational case
            for example in examples:
                prepare_multimodal_messages(example["messages"], len(example["images"]), len(example["videos"]))
            messages = [example["messages"] for example in examples]
            texts = self.processor.apply_chat_template(messages)
        elif self.dataset_text_field in examples[0]:  # standard case
            texts = [example[self.dataset_text_field] for example in examples]
        else:
            raise KeyError("The input examples must contain either 'messages' for conversational data or 'text' for standard " "data.")

        output = self.processor(
            images=images,
            text=texts,
            videos=videos,
            padding=True,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        labels = output["input_ids"].clone()
        labels[output["attention_mask"] == 0] = -100

        for visual_token_id in self.visual_token_ids:
            labels[labels == visual_token_id] = -100
        
        output["labels"] = labels
        return output

def main():
    # args = parse_args()
    # set_seed(args.seed)

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))


    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    set_seed(training_args.seed)


    if training_args.report_to == "none":
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = "MMEB-SFT"
        if training_args.run_name:
            os.environ["WANDB_NAME"] = training_args.run_name
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print0(">>> Loading model & processor ...")
    if 'Qwen2-VL' in model_args.model_name:
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        processor = Qwen2VLProcessor.from_pretrained(model_args.model_name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28, trust_remote_code=True)
        model_class = Qwen2VLForConditionalGeneration
    elif 'Qwen2.5-VL' in model_args.model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28, trust_remote_code=True)
        model_class = Qwen2_5_VLForConditionalGeneration
    # elif 'InternVL' in model_args.model_name:
    #     from transformers import AutoModelForImageTextToText, AutoProcessor
    #     processor = AutoProcessor.from_pretrained(model_args.model_name, trust_remote_code=True)
    #     model_class = AutoModelForImageTextToText
    # TODO: how to do mixed image-video training for InternVL?
    elif "Perception-LM" in model_args.model_name:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_class = AutoModelForImageTextToText
    else:
        raise NotImplementedError(f"Model {model_args.model_name} is not supported yet.")

    model = model_class.from_pretrained(
        model_args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    peft_config = None
    model = model_class.from_pretrained(
        model_args.model_name,
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config,
        # low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    if model_args.lora:
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            init_lora_weights="gaussian",
            use_dora=True,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[x.strip() for x in model_args.lora_target_modules.split(",") if x.strip()],
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if "qwen" in model_args.model_name.lower():
        model.model.visual.requires_grad_(False)
    elif "internvl" in model_args.model_name.lower():
        model.model.vision_tower.requires_grad_(False)

    tok = processor.tokenizer
    tok.padding_side = "right"

    warmup_ratio = 0.03 if model_args.lora else 0.1
    max_grad_norm = 0.3 if model_args.lora else 1.0

    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    training_args = SFTConfig(
        deepspeed=None if model_args.lora else { "train_batch_size": "auto",
                                                        "train_micro_batch_size_per_gpu": "auto",
                                                        "gradient_accumulation_steps": "auto",
                                                        "zero_optimization": {
                                                            "stage":1
                                                        }},
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        learning_rate=training_args.learning_rate,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        bf16=True,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        seed=training_args.seed,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        remove_unused_columns=False,
        gradient_checkpointing=not model_args.lora,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=training_args.dataloader_num_workers,
        report_to=None if training_args.report_to == "none" else training_args.report_to,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        save_safetensors=True,
        average_tokens_across_devices=False,
        save_total_limit=1
    )

    with training_args.main_process_first(local=False):
        os.makedirs(data_args.cache_dataset_dir, exist_ok=True)

        train_ds = init_sft_dataset(
            dataset_config=data_args.dataset_config,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            processor=processor
        )
        train_ds.set_transform(train_transform)

    n_samples = train_ds.num_rows if hasattr(train_ds, "num_rows") else len(train_ds)
    print0(f">>> Training set ready: {n_samples} samples")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        peft_config=peft_config,
        data_collator=MultimodalCollator(processor,
                                         visual_token_ids=[processor.image_token_id,
                                                           processor.video_token_id]),
    )
    trainer.train()

    if is_main_process():

        if model_args.lora:
            print0(">>> Merging LoRA weights into the base model ...")
            model = model.merge_and_unload()
            os.makedirs(training_args.output_dir + "_merged", exist_ok=True)
            model.save_pretrained(training_args.output_dir + "_merged")
            processor.save_pretrained(training_args.output_dir + "_merged")
        else:
            print0(f">>> Model checkpoints and processor saved to {training_args.output_dir}")
            processor.save_pretrained(training_args.output_dir)
            trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
