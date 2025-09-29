import logging

import PIL
from transformers.image_utils import ChannelDimension

# from src.model.baseline_backbone.colpali import ColPaliProcessor

logger = logging.getLogger(__name__)

import torch
import numpy as np
from src.utils import print_master

# from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration
# from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
# from src.model.vlm_backbone.qwen2_vl_tokenselection import \
#     Qwen2VLForConditionalGeneration as Qwen2VLTokenSelectionForConditionalGeneration, \
#     Qwen2VLProcessor as Qwen2VLTokenSelectionProcessor
# from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import InternVideo2_Stage2
from src.model.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
# from src.model.vlm_backbone.qwen2_5_vl_tokenselection import \
#     Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_TokenSelectionForConditionalGeneration
from transformers import AutoModelForImageTextToText

PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl_tokenselection'
QWEN2_5_VL_TOKENSELECTION = 'qwen2_5_vl_tokenselection'
INTERNVIDEO2 = 'internvideo2'
INTERNVL3 = "internvl"
GME = 'gme'  # QWEN2-VL
LamRA = 'lamra'  # QWEN2-VL
LamRA_QWEN2_5 = 'lamra_qwen25'  # QWEN2.5-VL
COLPALI = 'colpali'  # PaliGemma-3B
E5_V = 'e5_v'  # Llava_next
PLM = 'perception_lm'
MODEL2BACKBONE = {  # keys are from hf_config.model_type or manually added if not provided
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_vl_tokenselection': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
    'qwen2_vl_tokenselection': QWEN2_VL_TOKENSELECTION,
    'qwen2_5_vl_tokenselection': QWEN2_5_VL_TOKENSELECTION,
    'internvideo2': INTERNVIDEO2,
    'gme': GME, 
    'lamra': LamRA,
    'lamra_qwen25': LamRA,
    'colpali': COLPALI,
    "internvl": INTERNVL3,
    'e5_v': E5_V,
    'perception_lm': PLM
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|image_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|image_pad|>",
    GME: "<|image_pad|>",
    LamRA: "<|image_pad|>",
    LamRA_QWEN2_5: "<|image_pad|>",
    INTERNVL3: "<image>",
    INTERNVIDEO2: "",
    COLPALI: "",
    E5_V: "<image>",
    PLM: "<|image_pad|>"
}

VLM_VIDEO_TOKENS = {
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|video_pad|>",
    QWEN2_5_VL: "<|video_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|video_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|video_pad|>",
    GME: "<|video_pad|>",
    LamRA: "<|video_pad|>",
    LamRA_QWEN2_5: "<|video_pad|>",
    INTERNVIDEO2: "",
    INTERNVL3: "<video>",
    COLPALI: "",
    E5_V: "<image>",
    PLM: "<|video_pad|>",
}

backbone2model = {
    # PHI3V: Phi3VForCausalLM,
    # LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
    # QWEN2_VL_TOKENSELECTION: Qwen2VLTokenSelectionForConditionalGeneration,
    # QWEN2_5_VL_TOKENSELECTION: Qwen2_5_VL_TokenSelectionForConditionalGeneration,
    # INTERNVIDEO2: InternVideo2_Stage2,
    # E5_V: LlavaNextForConditionalGeneration,
    INTERNVL3: AutoModelForImageTextToText,
    PLM: AutoModelForImageTextToText
}

def get_visual_token_ids(processor):
    if "Qwen2VLProcessor" in str(processor.__class__):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    return image_tokens


def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    Note: due to this change, https://github.com/huggingface/transformers/commit/9215cc62d4366072aacafa4e44028c1ca187167b#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356L1102
    """
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name

    padding_side = "right" if model_args.do_sft_query else "left"
    print_master(f'Loading processor from: {model_name_or_path}')
    if model_args.model_backbone == PHI3V:
        from src.model.baseline_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = padding_side
    elif model_args.model_backbone == LLAVA_NEXT:
        from src.model.baseline_backbone.llava_next import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
    elif model_args.model_backbone in [QWEN2_VL, GME, LamRA]:
        from src.model.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path, padding_side=padding_side)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer, size=size
        )
    elif model_args.model_backbone in [QWEN2_5_VL, LamRA_QWEN2_5]:
        from src.model.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        # image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path, padding_side=padding_side)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, image_processor=image_processor, tokenizer=tokenizer)

        # from transformers import Qwen2_5_VLProcessor
        # pixel_kwargs = {}
        # if data_args is not None:
        #     if data_args.resize_min_pixels:
        #         pixel_kwargs['min_pixels'] = data_args.resize_min_pixels
        #     if data_args.resize_max_pixels:
        #         pixel_kwargs['max_pixels'] = data_args.resize_max_pixels
        # processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, **pixel_kwargs)
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            padding_side=padding_side
        )
    return processor


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Qwen2_VL_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None, **kwargs):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]

    # 1. iterate each pair and process, since processors do not support processing for mixed batch (contains data w/ and w/o visual inputs)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            try:
                if vlm_image_token in text:
                    if isinstance(images, PIL.Image.Image):
                        # images is a single image
                        images = [images]
                    for iid, image in enumerate(images):
                        # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                        if image.size[0] < 28 or image.size[1] < 28:
                            image = image.resize((56, 56))
                            images[iid] = image
                    inputs = processor(text=[text], images=images, return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
                elif vlm_video_token in text:
                    # TODO: check text/video data validity
                    inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
                else:
                    raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in the text: {text}")
            except Exception as e:
                # for i in images:
                #     print(i.filename)
                raise e
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

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
        'images': visual_inputs,
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    return inputs


def process_fn(model_inputs: dict, processor, model_backbone=None, **kwargs):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    vlm_image_token, vlm_video_token = processor.image_token, processor.video_token
    
    image_inputs, video_inputs = [], []

    for text, visual in zip(texts, visual_inputs):
        if visual is None or (type(visual)==list and any(i is None for i in visual)):
            # all images must be valid
            pass
        else:

            if vlm_image_token in text:
                visual = visual if isinstance(visual, PIL.Image.Image) else visual[0]
                if visual.size[0] < 28 or visual.size[1] < 28:
                    visual = visual.resize((56, 56))
                image_inputs.append(visual)
            elif vlm_video_token in text:
                assert isinstance(visual, list) and len(visual) > 1, f"Video data must have more than 1 frame, got {type(visual)} with length {len(visual) if isinstance(visual, list) else 'N/A'}"
                video_inputs.append(visual)
            else:
                raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in the text: {text}")

    inputs = processor(text=texts,
                       images=image_inputs if image_inputs else None,
                       videos=video_inputs if video_inputs else None,
                       padding=True,
                       padding_side="left")

    return inputs


process_vlm_inputs_fns = {
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL: Qwen2_VL_process_fn,
    # QWEN2_5_VL: process_fn,
    INTERNVL3: process_fn

}


def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):

    prompt = instruction
    if text:
        prompt = prompt + " " + text
    if add_video_token:
        video_token = VLM_VIDEO_TOKENS[model_backbone]
        prompt = video_token + " " + prompt
    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]
        prompt = image_token + " " + prompt

    return prompt