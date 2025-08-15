import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor, AutoConfig, AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.eval_utils import get_pred
from src.utils import print_rank
from src.model_utils import get_backbone_name
# from internvl_inference import split_model, load_image
from qwen_inference import batch_inference as batch_inference_qwen
import re
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
import datasets
from lmdeploy.vl.constants import IMAGE_TOKEN
from datasets import load_from_disk

instruction_prompt_v1 = """<image>\nPlease describe the image in details that can be used to retrieve similar images from a large image database. Please adjust your focus based on the main content. For instance, if the image is about an animal, you should describe the animal type, color, size, surroundings etc. If the image is a document or with texts, you should describe the document type, text content, language, etc. Please be as succinct as possible. Do not use any special characters besides commas, just use plain text. The description should be concise and informative while being succinct (you do not need to describe in a fluent and narrative way), with a focus on key, unique, and distinct visual attributes, for which the retrieved contents should also share.""" #. You do not need to describe the image in a narrative way, just list the attributes in a concise way. Do not use any special characters or punctuation marks, just use plain text."""

# prompts_base = {
#     "i2t": """Please describe the image in details that can be used to retrieve similar images' captions from a large image database. Please adjust your focus based on the main content. For instance, if the image is about an animal, you should describe the animal type, color, size, surroundings etc. If the image is a document or with texts, you should describe the document type, text content, language, etc. Do not use any special characters besides commas, just use plain text. The description should be concise and informative while being succinct (you do not need to describe in a fluent and narrative way), with a focus on key, unique, and distinct visual attributes including key colors and textures, for which the retrieved contents should also possess. Make sure to mention anything that is unique.""", #. You do not need to describe the image in a narrative way, just list the attributes in a concise way. Do not use any special characters or punctuation marks, just use plain text."""
#     "i2i": """Please describe the image in details that can be used to retrieve similar images from a large image database. Please adjust your focus based on the main content. For instance, if the image is about an animal, you should describe the animal type, color, size, surroundings etc. If the image is a document or with texts, you should describe the document type, text content, language, etc. Do not use any special characters besides commas, just use plain text. The description should be concise and informative while being succinct (you do not need to describe in a fluent and narrative way), with a focus on key, unique, and distinct visual attributes including key colors and textures, for which the retrieved contents should also share. Make sure to mention anything that is unique.""",
# }

prompts = {

    "InfographicsVQA": "Given the provided image about infographics, answer the following question in a single word or phrase: \n\n{}",

    "SUN397": "Given the provided image, what is the most likely scene category? Please answer with a single word or phrase: ",

    "VOC2007": "What is the main object category shown in the given image? Please answer with a single word or phrase: ",

    "ImageNet_1K": "What is the main object category shown in the given image? Please answer with a single word or phrase: ",

    "HatefulMemes": "Does the text in the provided image contain hateful content? Please answer yes or no: ",

    "VisDial": 
"""The below dialogue is about an image:

{}

Generate a caption of the image based on the dialogue. The caption should be concise, informative, but faithful to the dialogue. Please include all information that can be deduced from the dialogue, but do not add additional information that you cannot conclude from the dialogue.""",
    "A-OKVQA": """Given the image, answer the following question with a single word or phrase: 

{}""",

    "OK-VQA": """Given the image, answer the following question with a single word or phrase: 

{}""",

    "ChartQA": """Given the provided chart image, answer the following question:

{}""",

    "Visual7W": """Given the image, answer the following question with a single word or phrase: 

{}""",

    "CIRR": """Given the base image and the modification instruction, generate a caption of the target image that matches the instruction. """

}

prompts_descriptions = {

    "InfographicsVQA": "Given the provided image about infographics, answer the following question in a single word or phrase: \n\n{}",

    "SUN397": "Given the provided image, what is the most likely scene category? Please answer with a single word or phrase: ",

    "VOC2007": "What is the main object category shown in the given image? Please answer with a single word or phrase: ",

    "ImageNet_1K": "What is the main object category shown in the given image? Please answer with a single word or phrase: ",

    "HatefulMemes": "Does the text in the provided image contain hateful content? Please answer yes or no: ",

    "VisDial": 
"""The below dialogue is about an image:

{}

Generate a caption of the image based on the dialogue. The caption should be concise, informative, but faithful to the dialogue. Please include all information that can be deduced from the dialogue, but do not add additional information that you cannot conclude from the dialogue.""",
    "A-OKVQA": """Given the image, answer the following question with a single word or phrase: 

{}""",

    "OK-VQA": """Given the image, answer the following question with a single word or phrase: 

{}""",

    "ChartQA": """Given the provided chart image, answer the following question:

{}""",

    "Visual7W": """Given the image, answer the following question with a single word or phrase: 

{}""",

    "CIRR": """Given the base image and the modification instruction, generate a caption of the target image that matches the instruction. """

}

prompts_cot_v1={
    "CIRR": """Given a base image and a modification instruction that describes how the base image should be modified to match with a target image, rigorously follow and include the below steps in your response:

First, generate a rigorous and succinct reasoning process including the following steps:
1. Describe the base image.
2. Detail how the base image should be modified according to the instruction.
3. Analyze which visual features should remain, be altered, or omitted in the target image.

Next, based on your reasoning, generate a concise, succinct and faithful COCO-style caption of the target image. The caption should focus on key, unique, and distinct visual attributes, especially those highlighted in the instruction and omit any unecessary details from the base image that do not need to be present in the target image. Start your reasoning process by prefix "### Reasoning:". Start your caption with prefix "### Caption:".

Modification instruction: {}
""",
    "OK-VQA":"""Given the image, answer the following question: 

{}

Please first generate a rigorous and succinct reasoning process of what the question is asking, and how do you derive the answer from the image, prefixed by "Reasoning:". Then provide a single answer in one or two words, prefixed by "Answer:".""",
   
    "HatefulMemes": """What text is shown in the provided image? Does the text include inappropriate content? Please first generate the text in your response, prefixed by "Text:", Then, answer with a single yes or no, prefixed by "Answer:". Please include both text and your answer in your response.""",
    "HatefulMemes2": """What text is shown in the provided image?

Please first generate the text in your response, prefixed by "Text:", 

Then, determine whether the text and the image include inappropriate content. Answer with a single yes or no, prefixed by "Answer:". 

Make sure to include both text (prefixed by "Text:") and your answer (prefixed by "Answer:") in your response."""
}

prompts_cot_v2 = {
    "OK-VQA": """Given the image, answer the following question:

{}

Please generate a rigorous and succinct reasoning process along with your answer.""",
    "A-OKVQA": """Given the image, answer the following question:

{}

Please generate a rigorous and succinct reasoning process along with your answer.""",
    "SUN397": """Given the image, identify the most likely scene category.

Please generate a rigorous and succinct reasoning process along with your answer.""",
    "A-OKVQA": """Given the image, answer the following question:

{}

Please generate a rigorous and succinct reasoning process along with your answer.""",
}

prompts_cot_v3 = {
    "OK-VQA": """Given the image, answer the following question:

{}

Please first generate a rigorous and succinct reasoning process of what the question is asking, and how do you derive the answer from the image. Then provide a single answer in one or two words.""",
    "A-OKVQA": """Given the image, answer the following question:

{}

Please generate a rigorous and succinct reasoning process along with your answer.""",
    "SUN397": """Given the image, identify the most likely scene category.

Please generate a rigorous and succinct reasoning process along with your answer.""",
    "A-OKVQA": """Given the image, answer the following question:

{}

Please generate a rigorous and succinct reasoning process along with your answer.""",
}

replace_texts = {
    "CIRR": "<|image_1|>\nGiven an image, find a similar everyday image with the described changes: ",
    "OK-VQA": "<|image_1|>\nRepresent the given image with the following question: ",
    "InfographicsVQA": "<|image_1|>\nRepresent the given image with the following question: ",
    "MSCOCO": "",
    "VisDial": "Represent the given dialogue about an image, which is used for image retrieval: "
}

prefix_keys = {
    "CIRR": {
        "OpenGVLab/InternVL3-38B": ("### Reasoning:", "### Caption:"),
        "Qwen/Qwen2-VL-2B-Instruct": ("### Reasoning:", "### Caption:"),
    },
    "OK-VQA": {
        "OpenGVLab/InternVL3-38B": ("### Reasoning:", "### Answer:"),
        "Qwen/Qwen2-VL-2B-Instruct": ("Reasoning:", "Answer:"),
    },
    "HatefulMemes": {
        "OpenGVLab/InternVL3-38B": ("### Text:", "### Answer:"),
        "Qwen/Qwen2-VL-2B-Instruct": ("Text:", "Answer:"),
    }
}

IMAGE = "image"
TEXT = "text"
VIDEO = "video"

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

def process_fn(qry, subset):
    if subset in {"CIRR"}:
        return qry.replace("<|image_1|>\nGiven an image, find a similar everyday image with the described changes: ", "").strip()
    elif subset in {"MSCOCO"}:
        return re.search(r'"([^"]*)"', qry).group(1).strip()
    elif subset in {"HatefulMemes", "VOC2007", "NIGHTS", "VisualNews_i2t", "SUN397", "MSCOCO_i2t", "ImageNet_1K"}:
        return None
    elif subset in {"A-OKVQA", "OK-VQA", "Visual7W", "InfographicsVQA", "ChartQA"}:
        return qry.replace("<|image_1|>\nRepresent the given image with the following question: ", "").strip()
    elif subset in {"VisualNews_t2i"}:
        return qry.replace("Retrieve an image of this news caption. ", "").strip()
    elif subset in {"MSCOCO_t2i"}:
        return qry.replace("Find me an everyday image that matches the given caption: ", "").strip()
    elif subset in {"WebQA"}:
        return qry.replace("<|image_1|>\nFind a Wikipedia image that answers this question: ", "").strip()
    elif subset in {"VisDial"}:
        return qry.replace("Represent the given dialogue about an image, which is used for image retrieval: ", "").strip()
    elif subset in {"N24News"}:
        return qry.replace("<|image_1|>\nRepresent the given news image with the following caption for domain classification: ", "").strip()


def main():
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
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    # batch_inference = batch_inference_internvl if 'internvl' in model_args.model_name.lower() else batch_inference_qwen

    # hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    # model_backbone = get_backbone_name(hf_config=hf_config)
    # setattr(model_args, 'model_backbone', model_backbone)
    # setattr(training_args, 'model_backbone', model_backbone)
    # print_rank(f'model_backbone: {model_backbone}')
    # print(f'model_args: {json.dumps(vars(model_args), indent=2)}')
    # print(f'data_args: {json.dumps(vars(data_args), indent=2)}')
    # model, tokenizer = load_model(model_args.model_name)
    # model.eval()
    # model = model.to(training_args.device, dtype=torch.bfloat16)

    pipe = pipeline(model_args.model_name, backend_config=TurbomindEngineConfig(session_len=8192, tp=1, output_last_hidden_state='generation'))

    # if data_args.prompt_format == 'v2':
    if data_args.prompt_format == 'cot':
        if data_args.prompt_version == 'v1':
            prompts = prompts_cot_v1
        elif data_args.prompt_version == 'v2':
            prompts = prompts_cot_v2
        elif data_args.prompt_version == 'v3':
            prompts = prompts_cot_v3

    else:
        prompts = prompts_base
    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(data_args.subset_name):
        
        prompt = prompts[subset]

        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        # if data_args.prompt_format == 'cot':
        #     reasoning_prefix, description_prefix = prefix_keys[subset][model_args.model_name]

        # dataset = load_from_disk(os.path.join(data_args.dataset_path, subset, data_args.split_name[0] + ".hf")) 
        dataset = load_dataset(data_args.dataset_name, subset, split=data_args.split_name[0])

        qry_image_field = "qry_image_path" if "qry_image_path" in dataset.column_names else "qry_img_path"
        qry_text_field = "qry" if "qry" in dataset.column_names else "qry_text"
        tgt_image_field = "tgt_img_path" if "tgt_img_path" in dataset.column_names else "pos_image_path"
        tgt_text_field = "tgt_text" if "tgt_text" in dataset.column_names else "pos_text"

        dataset = dataset.to_pandas()
        dataset = dataset.drop_duplicates(subset=[qry_text_field, qry_image_field, tgt_text_field, tgt_image_field])
        dataset = datasets.Dataset.from_pandas(dataset)
        if data_args.n_partitions > 1:
            dataset = dataset.shard(num_shards=data_args.n_partitions, index=data_args.current_partition-1)

        image_folder = data_args.image_dir
            
        print(f"Processing {len(dataset)} images in {subset} for partition {data_args.current_partition}/{data_args.n_partitions}")
        folder = os.path.join("descriptions",  subset, model_args.model_name.split("/")[-1], data_args.prompt_version, data_args.prompt_format)
        descriptions = pickle.load(open(data_args.existing_desc, "rb")) if data_args.existing_desc and os.path.exists(data_args.existing_desc) else {}

        descriptions = {}

        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), training_args.per_device_eval_batch_size), desc="expand query"):

                batch = dataset[i:i + training_args.per_device_eval_batch_size]

                qry_texts, qry_images, tgt_images, tgt_texts = batch[qry_text_field], batch[qry_image_field], batch[tgt_image_field], batch[tgt_text_field]

                queries = [process_fn(qry, subset) for qry in qry_texts]

                tgt_images = [x[0] for x in tgt_images] if isinstance(tgt_images[0], list) else tgt_images
                tgt_texts = [x[0] for x in tgt_texts] if isinstance(tgt_texts[0], list) else tgt_texts

                loaded_qry_images = [load_image(os.path.join(image_folder, qry_image)) for qry_image in qry_images]
                inputs = [(prompt.format(q), qry_image) for q, qry_image in zip(queries, loaded_qry_images)]

                responses = [x for x in pipe(inputs, GenerationConfig(do_sample=True))]

                for qry, qry_img_path, response in zip(qry_texts, qry_images, responses):
                    descriptions[(qry, qry_img_path)] = response
                
        os.makedirs(folder, exist_ok=True)
        pickle.dump(descriptions, open(os.path.join(folder, f"descriptions_{data_args.dataset_split}_{data_args.current_partition}-{data_args.n_partitions}.pkl"), "wb"))

if __name__ == "__main__":
    main()
