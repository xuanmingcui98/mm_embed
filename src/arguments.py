from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name: str = field(
        metadata={"help": "huggingface model name or path"}
    )
    model_backbone: str = field(
        default=None,
        metadata={"help": "backbone name"}
    )
    processor_name: str = field(
        default=None, metadata={"help": "processor_name, huggingface model name or path"}
    )
    model_type: str = field(
        default=None, metadata={"help": "lavis model type"}
    )
    checkpoint_path: str = field(
        default=None, metadata={"help": "a local model path"}
    )
    sft_checkpoint_path: str = field(
        default=None, metadata={"help": "a local model path for supervised fine-tuning-based VLM"})
    pooling_module: str = field(
        default='last',
        metadata={"help": "pooling method for encoder"}
    )
    normalize: bool = field(
        default=True,
        metadata={"help": "normalize query and passage representations"}
    )
    temperature: float = field(
        default=0.02,
        metadata={"help": "temperature for softmax"}
    )
    lora: bool = field(
        default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "lora r"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "lora alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"}
    )
    lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
        metadata={"help": "lora target modules"}
    )
    num_crops: int = field(
        default=16,
        metadata={"help": "number of crops used in image encoder"}
    )
    prompt: str = field(
        default=None,
        metadata={"help": "prompt for the model'"}
    )
    freeze_base_model: bool = field(
        default=False,
        metadata={"help": "freeze base model"}
    )
    intra_llm_batch_size: int = field(
        default=16, metadata={"help": "intra-llm batch size"}
    )
    do_sft: bool = field(
        default=False, metadata={"help": "do supervised fine-tuning"}
    )
    do_cl: bool = field(
        default=True, metadata={"help": "do contrastive learning"}
    )
    do_sft_target: bool = field(
        default=False, metadata={"help": "do supervised fine-tuning"}
    )
    pooling_n_queries: int = field(
        default=1, metadata={"help": "number of queries for pooling"}
    )
    pooling_n_heads: int = field(
        default=8, metadata={"help": "number of heads for pooling"}
    )
    mask_visual_tokens_for_pooling: bool = field(
        default=False, metadata={"help": "mask visual tokens for pooling"}
    )
    num_pooling_layers: int = field(
        default=4, metadata={"help": "number of pooling layers"}
    )
    pooling_last_n_layers: int = field(
        default=None, metadata={"help": "number of last layers to use for pooling"}
    )
    learnable_queries: int = field(
        default=None, metadata={"help": "number of learnable queries for pooling"}
    )

    sft_model_checkpoint_path: str = field(
        default=None, metadata={"help": "a local model path for supervised fine-tuning-based VLM. Different from sft_checkpoint_path, this is only for generation and will not be merged with the main model."}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    dataset_path: str = field(
        default=None, metadata={"help": "huggingface dataset path"}
    )
    split_name: List[str] = field(
        default='original', metadata={"help": "'original', 'diverse_instruction'"}
    )
    subset_name: List[str] = field(
        default=None, metadata={"help": "Useful for datasets with subsets"}
    )
    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )
    num_sample_per_subset: int = field(
        default=100, metadata={"help": "number of training samples per subset"}
    )
    image_dir: str = field(
        default=None, metadata={"help": "Image directory path"}
    )
    max_len: int = field(
        default=10000, metadata={"help": "The maximum total input sequence length after tokenization. "
                                        "Use with caution, since it may truncate text prompts due to large image lengths."},
    )
    embedding_type: str = field(
        default="", metadata={"help": "embedding type"}
    )
    image_resolution: str = field(
        default='high', metadata={"help": "for models i.e. LLaVA-next and Qwen, resize images first"}
    )
    descriptions: str= field(
        default=None, metadata={"help": "path to the descriptions file"}
    )
    perturb_gt_rate: float = field(
        default=0.0, metadata={"help": "rate of perturbing ground truth descriptions"}
    )
    current_partition: int = field(
        default=1, metadata={"help": "current partition for distributed training"}
    )
    n_partitions: int = field(
        default=1, metadata={"help": "number of partitions for distributed training"}
    )
    add_description_to_tgt: bool = field(
        default=False, metadata={"help": "add description to target text"}
    )
    prompt_version: str = field(
        default="v1", metadata={"help": "prompt version, base or cot"}
    )
    prompt_format: str = field(
        default="gt_only", metadata={"help": "prompt format, v1 or v2"}
    )
    max_desc_len: int = field(
        default=1024, metadata={"help": "maximum description length"}
    )

    description_dir: str = field(
        default=None, metadata={"help": "directory for saving descriptions"}
    )
    last_n_hidden_states: int = field(
        default=1, metadata={"help": "number of last hidden states to use for training"}
    )
    apply_chat_template: bool = field(
        default=True, metadata={"help": "apply chat template to the dataset"}
    )
    apply_chat_template_target: bool = field(
        default=False, metadata={"help": "apply chat template to the dataset"}
    )
    add_question_to_tgt: bool = field(
        default=False, metadata={"help": "add question to target text"}
    )
    use_default_system_prompt: bool = field(
        default=False, metadata={"help": "use default system prompt for the model"}
    )
    target_description_dir: str = field(
        default=None, metadata={"help": "directory for target descriptions"}
    )
    no_description_for_text_only: bool = field(
        default=False, metadata={"help": "do not use description for text-only queries"}
    )



@dataclass
class TrainingArguments(TrainingArguments):
    image_encoder_freeze: bool = field(
        default=False, metadata={"help": "huggingface model name"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "directory for saving trained models"}
    )
    project_name: str = field(
        default=None, metadata={"help": "project name"}
    )

    logging_steps: int = field(
        default=1, metadata={"help": "logging steps"}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "number of training epochs"}
    )
    grad_cache: bool = field(
        default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(
        default=2, metadata={"help": "query side subset size"})
    gc_p_chunk_size: int = field(
        default=2, metadata={"help": "target side subset size"})
    pooler_learning_rate: float = field(
        default=None, metadata={"help": "intra-llm batch size"}
    )
    cl_loss_scalar: float = field(
        default=1, metadata={"help": "scalar for contrastive loss"}
    )
    sft_loss_scalar: float = field(
        default=1, metadata={"help": "scalar for contrastive loss"}
    )
    random_seed: int = field(
        default=42
    )

    


@dataclass
class MTEBArguments:
    task_types: List[str] = field(
        default=None, metadata={"help": ""}
    )
    tasks: List[str] = field(
        default=None, metadata={"help": ""}
    )
