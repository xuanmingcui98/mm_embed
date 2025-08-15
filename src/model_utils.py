import logging
import torch
import numpy as np
from src.utils import print_master

from src.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from src.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
# from transformers import Qwen2_5_VLForConditionalGeneration
from src.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
from transformers import (Qwen2VLProcessor, 
                          AutoModelForImageTextToText,
                          AutoProcessor
                          )
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
INTERNVL3 = "internvl"
MODEL2BACKBONE = {  # keys are from hf_config.model_type
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
    'internvl': INTERNVL3,  
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

vlm_image_tokens = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
    INTERNVL3: "<|image_pad|>"
}

backbone2model = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
    INTERNVL3: AutoModelForImageTextToText
}

def get_visual_token_ids(processor):
    if "Qwen2VLProcessor" in str(processor.__class__):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    return image_tokens

def load_processor(model_args):
    """
    Load processor based on VLM backbone.
    """
    print_master('Loading processor')
    model_name = model_args.processor_name if model_args.processor_name else model_args.model_name
    if model_args.model_backbone == PHI3V:
        from src.vlm_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == LLAVA_NEXT:
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            trust_remote_code=True
        )
    elif model_args.model_backbone == QWEN2_VL:
        from src.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name,
            image_processor=image_processor, tokenizer=tokenizer,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
    elif model_args.model_backbone == QWEN2_5_VL:
        
        from src.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        # image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name)
        # tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        # processor = Qwen2_5_VLProcessor.from_pretrained(model_name, image_processor=image_processor, tokenizer=tokenizer)
        # processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True,)
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
    elif model_args.model_backbone == INTERNVL3:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True,
                                                  min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config):
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(images=image, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # dummy image inputs based on the first valid data point
        pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
        image_size_for_padding = torch.from_numpy(list(v for v in image_sizes if v is not None)[0])
        # make the batch full tensors
        pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)
        image_sizes = [torch.from_numpy(v) if v is not None else image_size_for_padding for v in image_sizes]
        image_sizes = torch.cat(image_sizes, dim=0)
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def format_data(sample, add_generation_prompt=False,use_default_system_prompt=True):
    if not use_default_system_prompt:
        # formatted_sample = [
        #     {"role": "system",
        #     "content": [{"type": "text", "text": sample['system_prompt']}],}
        # ]
        raise NotImplementedError("Custom system prompt is removed.")
    else:
        formatted_sample = [
            {"role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],}
        ]
    user_content = [] if sample['image'] is None else [{"type": "image", "image": sample['image']}]
    user_content.append({"type": "text", "text": sample['text']})
    formatted_sample.append({"role": "user", "content": user_content})


    # add the generation no matter if description is None or not
    # if sample['description'] is not None:
    if not add_generation_prompt:
        formatted_sample.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample['description'] if sample['description'] is not None else ""}],
        })
    return formatted_sample


def Qwen2_VL_process_fn(model_inputs: dict, processor, max_length=None, apply_chat_template=False, do_sft=False, add_generation_prompt=False, use_default_system_prompt=False):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    if apply_chat_template:
        texts = [processor.apply_chat_template(
            format_data({
            'text': text,
            'image': image,
            'description': description
            }, 
            add_generation_prompt = add_generation_prompt, #not add_generation_prompt, # and description is not None,  #TODO: this is hardcoded to False for evaluating previously trained model.
            use_default_system_prompt=use_default_system_prompt), 
            tokenize=False, add_generation_prompt=add_generation_prompt)
        for text, image, description in zip(texts, images, model_inputs['description'])]
    
        if not add_generation_prompt:
            texts = [text.strip() for text in texts]
            # texts = [text[:-10] for text in texts]  # remove the last 10 characters (eos).
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(images=[image], text=[text], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'texts': texts,
        'images': images,
    }

    if do_sft:
        labels = inputs['input_ids'].clone()

        labels[labels == processor.tokenizer.pad_token_id] = -100


    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
        pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
        pixel_values = torch.stack(pixel_values, dim=0)
        # image_grid_thw = np.concatenate(image_grid_thw, axis=0)
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_grid_thw'] = image_grid_thw

        if do_sft:
            image_tokens = get_visual_token_ids(processor)

            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100
    # else:
    #     inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
    #     inputs['image_grid_thw'] = [None] * input_ids.shape[0]
    if do_sft:
        inputs['labels'] = labels
    return inputs


def InternVL3_process_fn(model_inputs: dict, processor, max_length=None, apply_chat_template=False, do_sft=False, add_description=False, add_generation_prompt=False, system_prompt_key='qry_system_prompt', use_default_system_prompt=False):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    if apply_chat_template:
        texts = [
            format_data({
            'text': text,
            'image': image,
            'system_prompt': system_prompt,
            'description': description
            }, add_description=add_description and description is not None, use_default_system_prompt=use_default_system_prompt)
        for text, image, system_prompt, description in zip(texts, images, model_inputs[system_prompt_key], model_inputs['description'])]
    
        inputs = processor.apply_chat_template(texts, add_generation_prompt=add_generation_prompt, return_tensors="pt",padding=True, return_dict=True, tokenize=True)
        inputs['texts'] = processor.apply_chat_template(texts, add_generation_prompt=add_generation_prompt, return_tensors="pt",padding=True, return_dict=True, tokenize=False)
        
        # inputs = processor(texts, images, return_tensors="pt", tokenize=True)

            # texts = [text[:-10] for text in texts]  # remove the last 10 characters (eos).

    inputs['images'] = images
    # if not do_sft:
    #     del inputs['labels']
    return inputs

process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,
    LLAVA_NEXT: Llava_NEXT_process_fn,
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL: Qwen2_VL_process_fn,
    INTERNVL3: InternVL3_process_fn
}
