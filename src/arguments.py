from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name: str = field(default="Qwen/Qwen2-VL-2B-Instruct", metadata={"help": "huggingface model name or path"})
    model_type: str = field(default=None, metadata={"help": "model type, typically includes in config file, but sometimes needs mannually add"})
    processor_name: str = field(default=None, metadata={"help": "processor_name, huggingface model name or path"})
    model_backbone: str = field(default=None, metadata={"help": "HF model type"})
    checkpoint_path: str = field(default=None, metadata={"help": "a local model path, could be a LoRA version"})
    pooling_module: str = field(default='last', metadata={"help": "pooling method for encoder"})
    normalize: bool = field(default=False, metadata={"help": "normalize query and passage representations"})
    temperature: float = field(default=0.02, metadata={"help": "temperature for softmax"})
    lora: bool = field(default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"})
    lora_r: int = field(default=16, metadata={"help": "lora r"})
    lora_alpha: int = field(default=64, metadata={"help": "lora alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "lora dropout"})
    lora_target_modules: str = field(default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj", metadata={"help": "lora target modules"})
    num_crops: int = field(default=16, metadata={"help": "number of crops used in image encoder"})
    uigraph_use: bool = field(default=False, metadata={"help": "Enable ui graph for token selection"})
    uigraph_diff: int = field(default=1, metadata={"help": "Pixel difference used for constructing ui graph for token selection"})
    uigraph_rand: bool = field(default=False, metadata={"help": "Enable random graph construction for token selection"})
    uimask_ratio: float = field(default=0.5, metadata={"help": "Specify the percentage of patch tokens to skip per component for token selection"})
    uimask_rand: bool = field(default=False, metadata={"help": "Enable random token selection instead of uniform selection"})
    lm_skip_layer: str = field(default='[1,28,0]', metadata={"help": "Specify the layers of the language model to skip for token selection"})
    vis_skip_layer: str = field(default='[1,32,0]', metadata={"help": "Specify the layers of the vision model to skip for token selection"})


    # added
    sft_checkpoint_path: str = field(default=None, metadata={"help": "a local model path for supervised fine-tuning-based VLM"})
    freeze_base_model: bool = field(
        default=False,
        metadata={"help": "freeze base model"}
    )
    do_sft_query: bool = field(default=False, metadata={"help": "do supervised fine-tuning"})
    do_cl: bool = field(default=True, metadata={"help": "do contrastive learning"})
    do_sft_target: bool = field(default=False, metadata={"help": "do supervised fine-tuning"})
    pooling_n_queries: int = field(default=4, metadata={"help": "number of queries for pooling"})
    pooling_n_heads: int = field(default=8, metadata={"help": "number of heads for pooling"})
    mask_visual_tokens_for_pooling: bool = field(default=False, metadata={"help": "mask visual tokens for pooling"})
    num_pooling_layers: int = field(default=4, metadata={"help": "number of pooling layers"})
    pooling_last_n_layers: int = field(default=None, metadata={"help": "number of last layers to use for pooling"})
    meta_queries: int = field(default=None, metadata={"help": "number of meta queries for the model, if set, it will add special tokens to the tokenizer and resize the embedding layer"})
    meta_queries_aggregate_type: str = field(default='mean', metadata={"help": "how to aggregate meta queries, mean, concat, late_interaction, attention_pooler"})
    inter_task_temperature: float = field(
        default=None, metadata={"help": "temperature for inter-task contrastive loss"}
    )
    learnable_temperature: bool = field(
        default=False, metadata={"help": "learnable temperature"}
    )


@dataclass
class DataArguments:
    dataset_config: str = field(default=None, metadata={"help": "yaml file with dataset configuration"})
    data_basedir: str = field(default=None, metadata={"help": "Expect an absolute path to the base directory of all datasets. If set, it will be prepended to each dataset path"})
    dataset_name: str = field(default=None, metadata={"help": "huggingface dataset name"})
    subset_name: List[str] = field(default=None, metadata={"help": "Useful for datasets with subsets"})
    dataset_split: str = field(default='train', metadata={"help": "dataset split"})
    num_sample_per_subset: int = field(default=None, metadata={"help": "number of training samples per subset"})
    image_dir: str = field(default=None, metadata={"help": "Image directory path"})
    encode_output_path: str = field(default=None, metadata={"help": "encode output path"})
    max_len: int = field(default=None, metadata={"help": "The maximum total input sequence length after tokenization. Use with caution, since it may truncate text prompts due to large image lengths."},)
    embedding_type: str = field(default="", metadata={"help": "embedding type"})
    image_resolution: str = field(default=None, metadata={"help": "for models i.e. LLaVA-next and Qwen, resize images first, none means using original image resolution. This is only works when `--resize_use_processor false`."})
    resize_use_processor: bool = field(default=True, metadata={"help": "Resize visual inputs insides processor, e.g. Qwen2VLImageProcessor, instead of by our code."})
    resize_min_pixels: int = field(default=28*28*4, metadata={"help": "The min pixels of the image to resize the image. This is only works when `--resize_use_processor true`."})
    resize_max_pixels: int = field(default=28*28*1280, metadata={"help": "The max pixels of the image to resize the image. This is only works when `--resize_use_processor true`."})
    image_decay_factor: float = field(default=None, metadata={"help": "The image decay factor for resizing temporal images"})
    num_hardneg: int = field(default=0, metadata={"help": "hard negative number"})

    # added

    prompt_format: str = field(default="cot_gt", metadata={"help": "cot_answer, cot_only, and answer_only"})
    max_desc_len: int = field(default=1024, metadata={"help": "maximum description length"})
    query_description_dir: str = field(default=None, metadata={"help": "directory for saving descriptions"})
    apply_chat_template: bool = field(default=True, metadata={"help": "apply chat template to the dataset"})
    target_description_dir: str = field(default=None, metadata={"help": "directory for target descriptions"})
    debug_prompt: bool = field(default=False, metadata={"help": "debug mode, will not use the dataset_config and will use the dataset_name and subset_name instead"})
    rewrites_for_mm_only: bool = field(default=False, metadata={"help": "only use rewrites for multi-modal input"})
    max_rewrite_len: int = field(default=500, metadata={"help": "maximum rewrite length"})
    num_shards: int = field(default=2, metadata={"help": "number of shards to split the dataset into, useful for debugging with smaller datasets or distributed training"})
    rebuild_cache: bool = field(default=False, metadata={"help": "rebuild the cache, otherwise will load from existing cache if available"})
    cache_dataset_dir: str = field(default=None, metadata={"help": "directory to save the processed dataset cache"})

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default=None, metadata={"help": "directory for saving trained models"})
    resume_from: str = field(default="none", metadata={"help": "`auto` will detect if any previous checkpoints should be resumed. or specify specific step of the checkpoint."})
    project_name: str = field(default=None, metadata={"help": "project name"})
    logging_steps: int = field(default=1, metadata={"help": "logging steps"})
    num_train_epochs: int = field(default=1, metadata={"help": "number of training epochs"})
    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=2, metadata={"help": "query side subset size"})
    gc_p_chunk_size: int = field(default=2, metadata={"help": "target side subset size"})
    interleave_stopping_strategy: str = field(default="all_exhausted", metadata={"help": "all_exhausted or first_exhausted"})
    interleave_batch_size: float = field(default=0, metadata={"help": "Specify mini-batch size to interleave data from multi-sources, 0/None means random sampling by examples, 1 means full batch."})

    # added

    pooler_learning_rate: float = field(
        default=None, metadata={"help": "intra-llm batch size"}
    )
    cl_loss_scalar: float = field(
        default=1, metadata={"help": "scalar for contrastive loss"}
    )
    sft_loss_scalar: float = field(
        default=1, metadata={"help": "scalar for contrastive loss"}
    )

    use_symmetric_loss: bool = field(
        default=False, metadata={"help": "use symmetric loss for contrastive loss"}
    )
    deepspeed_stage: int = field(
        default=1, metadata={"help": "deepspeed zero optimization stage"}
    )

@dataclass
class MTEBArguments:
    device: str = field(default="cuda", metadata={"help": "use cuda for single GPU inference, if multiple GPUs are available it will use DP automatically"})
    batch_size_per_device: int = field(default=16, metadata={"help": ""})
    max_length: int = field(default=512, metadata={"help": ""})
    eval_output_dir: str = field(default=None, metadata={"help": "directory for saving trained models"})
    task_types: List[str] = field(default=None, metadata={"help": ""})
    tasks: List[str] = field(default=None, metadata={"help": ""})
    prompt_family: List[str] = field(default=None, metadata={"help": ""})
