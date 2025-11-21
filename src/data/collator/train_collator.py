from itertools import repeat
from typing import Optional
from torch.jit import isinstance
from typing import Any, Dict
import logging
import numpy as np
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from ...arguments import DataArguments, ModelArguments, TrainingArguments
import torch
from qwen_vl_utils import smart_resize
from io import BytesIO
from ...model.processor import LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, \
    QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, PHI3V, process_vlm_inputs_fns
from PIL import Image
import io
from ...utils import print_rank, print_master

logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000


def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):

    def chunk_dict(d):
        keys = list(d.keys())
        chunked_tensors = []
        for k in keys:
            if isinstance(d[k], torch.Tensor):
                chunked_tensor = d[k].split(chunk_size, dim=0)
            else:
                chunked_tensor = [d[k][i: i + chunk_size] for i in list(range(0, len(d[k]), chunk_size))]
            chunked_tensors.append(chunked_tensor)
        chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]
        chunked_inputs = [{arg_key: c} for c in chunked_arg_val]
        return chunked_inputs

    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    if type(arg_val) == dict:
        chunked_inputs = chunk_dict(arg_val)
    else:
        # hard negative case
        chunked_inputs_transposed = []
        for hn in arg_val:
            chunked_inputs_transposed.append(chunk_dict(hn))
        
        chunked_inputs = []
        for i in range(len(chunked_inputs_transposed[0])):
            chunked_inputs.append({arg_key: [x[i][arg_key] for x in chunked_inputs_transposed]})

    return chunked_inputs


def split_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]
    keys = list(arg_val.keys())

    # for input_ids and attention_mask, split directly
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in ["input_ids", "attention_mask"]]

    # for pixel_values and image_sizes, need to split based on the position of images
    input_ids = arg_val["input_ids"]
    # positions = torch.nonzero(((input_ids < 0) & (input_ids > -MAX_INPUT_ID)) | input_ids == LLAVE_IMAGE_TOKEN_ID, as_tuple=True)
    positions = torch.nonzero((input_ids < 0) & (input_ids > -PHI_IMAGE_TOKEN_MAX_INPUT_ID), as_tuple=True)
    row_contain_image = torch.unique(positions[0])  # indicates which row in input_ids contain images
    num_chunks = len(chunked_tensors[0])
    chunk_image_count = []
    for chunk_idx in range(num_chunks):
        chunk_image_count.append(torch.sum(
            (row_contain_image >= chunk_idx * chunk_size) & (row_contain_image < (chunk_idx + 1) * chunk_size)).item())
    if "pixel_values" in keys:
        pixel_values = arg_val["pixel_values"]
        image_sizes = arg_val["image_sizes"]
        chunked_tensors.append(torch.split(pixel_values, chunk_image_count))
        chunked_tensors.append(torch.split(image_sizes, chunk_image_count))

    chunked_arg_val = []
    for kk, tt in zip(repeat(keys), zip(*chunked_tensors)):
        if "pixel_values" in keys and tt[2].numel() == 0:  # this chunk doesn't contain image
            chunked_arg_val.append(dict(zip(kk[:2], tt[:2])))
        else:
            chunked_arg_val.append(dict(zip(kk, tt)))

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    """
    Get either qry_reps or tgt_reps or neg.
    """
    if x["qry_reps"] is not None:
        return x["qry_reps"]
    elif x['tgt_reps'] is not None:
        return x["tgt_reps"]
    elif x['neg_reps'] is not None:
        return x['neg_reps']
    else:
        raise ValueError("All values are none")


@dataclass
class TrainTextImageDataCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        neg_inputs = self._get_batch_inputs(examples, "neg_text", "neg_image")
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_keyname, image_keyname):
        texts, images = [], []
        for example in examples:
            # @ruimeng filter invalid data examples here will lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                text, image = '  ', None
            text, image = example[text_keyname], example[image_keyname]
            if type(text) == list:
                if len(text) == 0 or len(image) == 0:
                    text, image = '  ', None
                else:
                    text, image = text[0], image[0]
            texts.append(text)
            images.append(image)
        inputs = {'text': texts, 'image': images}
        return inputs

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

@dataclass
class Qwen2_5VLMultimodalProcessor:

    processor: ProcessorMixin
    max_length: Optional[int] = None
    completion_only_loss: bool = False  # default not used in practice; SFTTrainer always passes the relevant value
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, examples):

        text = examples['text']
        images = examples['images']
        videos = examples['videos']
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None

        vlm_image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        fake_visual_token_len = 0

        has_images = images is not None and any(i is not None for i in images)
        has_videos = videos is not None and any(v is not None for v in videos)

        if not (has_images or has_videos):
            text[0] = f"{vlm_image_token}{text[0]}"
            images = [Image.new("RGB", (32, 32), (255, 255, 255))]
            fake_visual_token_len = len(
                self.processor(text=vlm_image_token, images=images)['input_ids'][0]
            )

        output = self.processor(
            text=text,
            images=images,
            videos=videos,
            padding=True,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )

        attention_mask = output["attention_mask"].long()
        if fake_visual_token_len > 0:
            seq_len = attention_mask.shape[1]
            content_len = int(attention_mask[0].sum().item())
            start = seq_len - content_len  # first non-pad token (left-padding)
            attention_mask[0, start:start + fake_visual_token_len] = 0

        output['attention_mask'] = attention_mask
        return output

def get_inputs(examples, text_key, visual_key):
    # return a dict of lists of text, image and video
    texts, images, videos = [], [], []
    for example in examples:
        texts.append(example[text_key])
        vis_input = example[visual_key]
        if vis_input and vis_input['bytes'][0]:
            visuals = [safe_open_image(x) for x in vis_input['bytes']]
        elif vis_input and vis_input['paths'][0]:
            visuals = [safe_open_image(x) for x in vis_input['paths']]
        else:
            visuals = [None]

        is_video = len(visuals) > 1

        if visuals[0]:
            if is_video:
                videos.append(visuals)
            else:
                images.append(visuals)

    return {"text": texts,
            "images": images,
            "videos": videos}

@dataclass
class ContrastiveDataCollator:

    processor: Any

    def __post_init__(self):
        self.processor = Qwen2_5VLMultimodalProcessor(self.processor)

    def __call__(self, examples):

        qry_inputs = self.processor(get_inputs(examples, "query_text", "query_image"))
        pos_inputs = self.processor(get_inputs(examples, "pos_text", "pos_image"))
        # neg_inputs = self.processor(get_inputs(examples, "neg_text", "neg_image"))

        qry_inputs['text'] = [e['query_text'] for e in examples]
        pos_inputs['text'] = [e['pos_text'] for e in examples]
        qry_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        pos_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        qry_inputs['task_id'] = torch.tensor([e['task_id'] for e in examples])
        pos_inputs['task_id'] = torch.tensor([e['task_id'] for e in examples])
        return qry_inputs, pos_inputs
        

@dataclass
class MultimodalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    batch_size: Optional[int] = None  # used to verify if a batch has invalid data

    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        for example in batch:
            # @ruimeng filter invalid data examples here may lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                text, visual_input = '  ', None
            else:
                text, raw_images = example[text_keyname], example[image_keyname]
                if type(raw_images) == dict:
                    visual_input = []
                    assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
                    num_images = len(raw_images['resolutions'])
                    for image_idx in range(num_images):
                        bytes = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                        path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                        image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                        if bytes is None and path is None:
                            image = None
                        elif bytes is not None:
                            # vidore, image inputs are already bytes
                            image = Image.open(io.BytesIO(bytes))
                        elif path is not None:
                            # mmeb/video datasets, lazy image loading and processing
                            with Image.open(path) as img:
                                image = img.convert("RGB")
                        else:
                            print_rank(f"\n{'=' * 50}\nsomething went wrong with a data point from {example['global_dataset_name']}, neither bytes or path is given. \n\t\tquery_text: {example.get('query_text')}")
                        if not self.data_args.resize_use_processor and image is not None and image_resolution:
                            image = image.resize(image_resolution)
                        if image is not None and (self.data_args.image_decay_factor is not None and image_resolution is None):
                            assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                            assert self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], "image_decay_factor is only supported for Qwen models"
                            # TODO: this is a hacky way to decay image resolution, need to be refactored
                            max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_max_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))
                            width, height = image.size
                            resized_height, resized_width = smart_resize(
                                height,
                                width,
                                min_pixels=self.data_args.resize_min_pixels,
                                max_pixels=max_pixels,
                            )
                            image = image.resize((resized_width, resized_height))  
                        visual_input.append(image)
                else:
                    visual_input = None
            texts.append(text)
            visual_inputs.append(visual_input)
        inputs = {'text': texts, 'images': visual_inputs}
        return inputs


    def __call__(self, examples):
        """
        :param examples: 'query_text', 'query_image_path', 'pos_text', 'pos_image_path', 'neg_text', 'neg_image_path'
        """
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")

        neg_inputs = []
        if self.data_args.hard_negative_dir:

            for idx in range(self.data_args.hard_negatives_per_sample):
                sample_neg_inputs = []
                for example in examples:
                    sample_neg_inputs.append({
                        "neg_text": example['neg_text'][idx],
                        "neg_image": example['neg_image'][idx],
                        "global_dataset_name": example['global_dataset_name']
                    })
                sample_neg_inputs = self._get_batch_inputs(sample_neg_inputs, "neg_text", "neg_image")
                neg_inputs.append(sample_neg_inputs)
        
        bs = len(qry_inputs['text'])
        assert bs > 0, 'An empty batch'
        # pad batch to batch_size to avoid hanging in distributed training
        if self.batch_size is not None and bs < self.batch_size:
        # if bs < self.training_args.per_device_train_batch_size:
            raise RuntimeError(f"Expect batch size {self.training_args.per_device_train_batch_size}, but got batch size of {bs}")

        process_fn = process_vlm_inputs_fns[self.training_args.model_backbone]
        qry_inputs = process_fn(qry_inputs, processor=self.processor, max_length=self.data_args.max_len, model_backbone=self.model_args.model_backbone)
        pos_inputs = process_fn(pos_inputs, processor=self.processor, max_length=self.data_args.max_len, model_backbone=self.model_args.model_backbone)
        neg_inputs = [process_fn(neg_input, processor=self.processor, max_length=self.data_args.max_len, model_backbone=self.model_args.model_backbone) for neg_input in neg_inputs]
        qry_inputs['text'] = [e['query_text'] for e in examples]
        pos_inputs['text'] = [e['pos_text'] for e in examples]
        qry_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        pos_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        qry_inputs['task_id'] = torch.tensor([e['task_id'] for e in examples])
        pos_inputs['task_id'] = torch.tensor([e['task_id'] for e in examples])
        qry_inputs['index_id'] = [e['index_id'] for e in examples]
        pos_inputs['index_id'] = qry_inputs['index_id']
        # print_rank(f"\t\tQry collator: qry_inputs['input_ids'].shape={qry_inputs['input_ids'].shape}\t\tPos collator: pos_inputs['input_ids'].shape={pos_inputs['input_ids'].shape}")
        return qry_inputs, pos_inputs, neg_inputs

def get_visual_token_ids(processor):
    if "Qwen2VLProcessor" in str(processor.__class__):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    return image_tokens