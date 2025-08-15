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
import re
import datasets
from datasets import load_from_disk
# from src.dataset import process_fn
import signal
from vllm import LLM, SamplingParams
import argparse
from PIL import Image

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']


task_categories = {
    "classification": {"ImageNet-1K", "ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"},
    "vqa": {"OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"},
    "retrieval": {"VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"},
    "grounding": {"MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"}
}

def process_fn(qry, subset):
    if subset in {"CIRR"}:
        return qry.replace("<|image_1|>\nGiven an image, find a similar everyday image with the described changes: ", "").strip()
    elif subset in {"FashionIQ"}:
        return qry.replace("<|image_1|>\nFind an image to match the fashion image and style note: ", "").strip()
    elif subset in {"EDIS"}:
        return qry.replace("<|image_1|>\nFind a news image that matches the provided caption: ", "").strip()
    elif subset in {"RefCOCO"}:
        return qry.replace("<|image_1|>\nSelect the portion of the image that answers the question ", "").strip()
    elif subset in {"Wiki-SS-NQ"}:
        return qry.replace("Find the document image that can answer the given query: ", "").strip()
    elif subset in {"OVEN"}:
        return qry.replace("<|image_1|>\nRetrieve a Wikipedia image-description pair that provides evidence for the question of this image: ", "").strip()
    elif subset in {"RefCOCO-Matching"}:
        return qry.replace("<|image_1|>\nSelect the portion of the image that follows the language expressions: ", "").strip() 
    elif subset in {"Visual7W-Pointing"}:
        return qry.replace("<|image_1|>\nSelect the portion of the image that answers the question ", "").strip()
    elif subset in {"MSCOCO"}:
        return re.search(r'"([^"]*)"', qry).group(1).strip()
    elif subset in task_categories["vqa"]:
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
    elif subset in task_categories["classification"] or subset in {"NIGHTS", "MSCOCO_i2t", "VisualNews_i2t"}:
        return None
    else:
        raise ValueError(f"Unknown subset: {subset}")

instruction_prompt_v1 = """<image>\nPlease describe the image in details that can be used to retrieve similar images from a large image database. Please adjust your focus based on the main content. For instance, if the image is about an animal, you should describe the animal type, color, size, surroundings etc. If the image is a document or with texts, you should describe the document type, text content, language, etc. Please be as succinct as possible. Do not use any special characters besides commas, just use plain text. The description should be concise and informative while being succinct (you do not need to describe in a fluent and narrative way), with a focus on key, unique, and distinct visual attributes, for which the retrieved contents should also share.""" #. You do not need to describe the image in a narrative way, just list the attributes in a concise way. Do not use any special characters or punctuation marks, just use plain text."""

# prompts_base = {
#     "i2t": """Please describe the image in details that can be used to retrieve similar images' captions from a large image database. Please adjust your focus based on the main content. For instance, if the image is about an animal, you should describe the animal type, color, size, surroundings etc. If the image is a document or with texts, you should describe the document type, text content, language, etc. Do not use any special characters besides commas, just use plain text. The description should be concise and informative while being succinct (you do not need to describe in a fluent and narrative way), with a focus on key, unique, and distinct visual attributes including key colors and textures, for which the retrieved contents should also possess. Make sure to mention anything that is unique.""", #. You do not need to describe the image in a narrative way, just list the attributes in a concise way. Do not use any special characters or punctuation marks, just use plain text."""
#     "i2i": """Please describe the image in details that can be used to retrieve similar images from a large image database. Please adjust your focus based on the main content. For instance, if the image is about an animal, you should describe the animal type, color, size, surroundings etc. If the image is a document or with texts, you should describe the document type, text content, language, etc. Do not use any special characters besides commas, just use plain text. The description should be concise and informative while being succinct (you do not need to describe in a fluent and narrative way), with a focus on key, unique, and distinct visual attributes including key colors and textures, for which the retrieved contents should also share. Make sure to mention anything that is unique.""",
# }

prompts_reasoning = {
    "MSCOCO_i2t":
"""Given an image and the ground-truth COCO-caption, describe briefly how the COCO-style caption can be derived from the image.

- First provide a detailed but succinct description of the image. 
- Then describe what a COCO-style caption should contain. (Hint: it should focus on the most salient objects and their arrangement in the image.)
- Wrap your reasoning in <think>...</think> (2–3 sentences).  
- Then write the final COCO caption on the next line as: Answer: <answer>

Importantly, do not reference the COCO-caption in your reasoning or answer. The COCO-caption is only provided for your reference. 

---

EXAMPLE

CAPTION: A young girl inhales with the intent of blowing out a candle.

<think>The image shows a young girl seated at a table, leaning toward a small bowl with a lit candle inside. Her mouth is wide open and her cheeks slightly drawn inward, a posture consistent with inhaling before blowing. The candle’s flame is steady, indicating the blowing hasn’t occurred yet. These visual cues suggest that the girl is preparing to blow out the candle. This pre-blow moment is the central action in the scene, which is typical in COCO-style captions that focus on key actions and objects in the image.</think>
Answer: A young girl inhales with the intent of blowing out a candle.

---

Now answer for the following image:

CAPTION: {query}
IMAGE: <image>""",

"CIRR": """Given a base image, a textual modification instruction, and the corresponding target image,  
reason about what the target image should likely look like **after applying the described change to the base image**.

You may use the target image for disambiguation or supervision, but **do NOT include any visual detail in your reasoning or answer unless it can be inferred from the base image and instruction**.

NOTES:
• The target image may differ significantly from the base.  
• Only the parts explicitly stated in the instruction should carry over.  
• Do NOT assume background, pose, or angle remain the same unless stated.  
• Wrap your reasoning in <think>...</think> (1–3 sentences).
• Then write a caption of the likely target image on the next line using: Answer: <answer>
- The caption should solely describe the target image. Do not refer to or compare with the base image or the instruction.

---

EXAMPLE

INSTRUCTION: Make the human facing the camera with less gear.

<think>The base image shows a scuba diver in profile, fully equipped with oxygen tank, wetsuit, and flippers, interacting with a jellyfish. The instruction asks to reduce the gear and turn the person to face the camera. This implies a swimmer with minimal equipment like a snorkel or goggles, shown from the front in a more casual or recreational pose.</think>
Answer: A swimmer facing the camera underwater with minimal gear, such as goggles and a snorkel.

---

Now complete the following:

BASE IMAGE: <image>  
TARGET IMAGE: <image>  
INSTRUCTION: {query}""",

    "MSCOCO": """Given an image, first describe the image in detail, then produce a COCO-style succinct caption. Follow the rules below:

- Generate a reasoning process (2–3 sentences) that describes the image in fine-grained details.
- Wrap your reasoning in <think>...</think>.
- Then output a COCO-style caption using: Answer: <answer>

---

EXAMPLE

<think>A large brown dog, likely a Bloodhound, is sitting on green grass in an outdoor park or showground. The dog has long, floppy ears, loose skin, and a prominent snout, which are characteristic of the breed. Its posture is upright but relaxed, and it's looking slightly to the side. Behind the dog, there are metal barricades and signs, as well as trees and tents, featuring a public outdoor setting.</think>
Answer: A brown Bloodhound sitting on grass in front of meta fence.

---

Now answer for the following query image:

IMAGE: <image>""",

    "NIGHTS": """Given a query image, use your background knowledge and visual reasoning to describe what kinds of visual features should be present in a matching target image. Your goal is to help a model retrieve a visually and contextually similar image based on salient elements. Follow the rules below:

- Generate a succinct reasoning process (2–3 sentences) that highlights the key visual or contextual features in the query image. This may include scene type, objects, structure, layout, lighting, textures, or mood.
- Wrap your reasoning in <think>...</think>.
- Then output the final description of the query image on the next line using: Answer: <answer>
- You are also given the target image for your reference. However, you should not directly describe the target image but rather focus on the query image and what a matching target image should look like.

Importantly, do not make reference to the target image in your reasoning or answer. The target image is only for your understanding of what a similar image should look like.

---

EXAMPLE

<think>The query image shows an outdoor restaurant scene at dusk, with modern architecture, visible indoor lighting, and set tables on a clean patio. To match it, the target image should depict a similar setting: an outdoor or semi-outdoor dining area, with aligned rows of tables, modern decor, prominent artificial lighting, and a warm or evening ambiance. The structure should feel open, inviting, and designed for dinner service.</think>
Answer: A modern outdoor dining area at dusk, featuring neatly arranged tables with white cloths, warm ambient lighting, and a stylish restaurant interior visible through large openings.

---

Now answer for the following query image:

QUERY IMAGE: <image>
TARGET IMAGE: <image>""",

    "WebQA": """Given the below question, the goal is to retrieve the relevant image-text pair that answers the question. 
    
Question: {query}

Now, please analyze the question, and generate a concise, detailed, while succinct description of what the target image and text ought to look like. 
You are also given the target image and text directly for your reference. However, as in reality the target image and text are not available, you should generate the description based on your knowledge and the question only. 
In other word, please pretend you did not see the target image and text while you generate the description. 

Target text: {target_text}
Target image: <image>

Please follow the below rules when generating description:
- Keep the generated description concise, detailed, while being succinct, within 1-2 sentences.
- Do not include every detail in the target image and text -- keep in mind they are only for your reference and will not be available in reality. You should **only** include information that can be strictly and reasonably derived from the question based on your knowledge.
- In other word, generate description as if you did not see the target image and text, but only based on the question and your background knowledge of what the target image and text ought to be about.""",
    "MSCOCO_t2i": """Given a caption, the goal is to describe what the target image may look like based on the caption and your background knowledge.

You are given the target image for your reference. However,you should only use the provided caption and your background knowledge to reasonably infer what the image may depict. The actual image is hidden — you must not describe any visual detail that is not strictly implied by the caption.

Target image: <image>

Instructions:
Use only the caption and general world knowledge to predict likely visual features.

Describe:

- the likely objects (e.g. people, animals, vehicles)
- their appearance or attributes (e.g. clothing, size, material, activity)
- the setting or environment (e.g. indoor/outdoor, beach, kitchen, street)
- Do not make up details not implied by the caption.

Be concise, yet specific and visual in your wording (1–2 sentences).

Think of it as guiding someone to imagine the image — but only using what is grounded in the caption.

Example:
Caption:

A man is riding a surfboard on a wave.

Generated visual description:

A man wearing swimwear is balancing on a surfboard amid ocean waves, likely in a beach or sea setting, with water splashing around him.

Now, generate a visual description for the following caption:

Caption:

{query}""",
#     "MSCOCO_t2i": """Given a caption, the goal is to retrieve the corresponding image that matches the caption.

# Now, please utilize your background knowledge to guess what the target image may look like based on the caption, and expand the caption to include more details about the visual features in the target image. You can make reasonable assumptions about the visual features in the image based on the caption. 

# You are also provided the target image for your reference. However, as in reality the target image is not available, you should generate the description based on your knowledge and the caption only. In other word, please pretend you did not see the target image while you generate the description.

# Caption: {query}

# Target image: {IMAGE_TOKEN}

# Generate a concise caption for the image following the below rules:
# - Be concise, detailed and succinct.
# - Do not include every detail in the target image -- you should **only** include visual features that can be strictly and reasonably derived from the given caption based on your knowledge.
# - In other word, please generate the caption as if you did not see the target image, but only based on the news text and your background knowledge. The target image is only provided for your reference, and should not be used to generate the caption.""",
    "VisualNews_t2i": """Given a news caption and its corresponding image, use your background knowledge and reasoning to expand the caption with more visual details that are likely to appear in the image. Use the image only for verification — do not include any visual elements unless they can be reasonably inferred from the caption.

Follow the instructions below:
- Generate a succinct reasoning process (2–3 sentences) about the likely visual scene described in the caption, using both your background knowledge and only what is justified by the text.
- You may refer to the image to help disambiguate uncertain textual references, but do **not** describe anything in the image that is not implied or suggested by the caption.
- Wrap your reasoning in <think>...</think>.
- Then output a final image description on the next line using: Answer: <answer>

---

EXAMPLE

CAPTION: Indian National Congress Vice President Rahul Gandhi addresses the special plenary session of Confederation of Indian Industry in New Delhi on April 4, 2013.

<think>Since it's a formal industry event, the setting is likely an indoor stage with branded backdrops. Rahul Gandhi is probably standing or speaking into a microphone. His attire is likely formal—possibly a white kurta. Background may show "CII" logos, and lighting is professional, consistent with a public address or keynote.</think>
Answer: Rahul Gandhi in a white kurta speaking into a microphone on a stage with CII logos in the background.

---

Now answer for the following caption and the reference target image:

CAPTION: {query}  
IMAGE: <image>
""",


    "OK-VQA": 
"""Given an image, a question, and the correct answer, explain step-by-step how the answer can be derived from the image. Please follow the below rules:
- Keep the reasoning concise and grounded in visual or factual evidence. Keep it succinct, within 1-2 sentences.
- Wrap your reasoning in <think> and </think> tags.  
- Then, on the next line, output the final answer, starting with "Answer:".
- The correct answer is provided for reference, but you should derive it from the image and question. Do not refer to the correct answer in your reasoning or answer.

---

Example:

QUESTION: What is the hairstyle of the blond called?
CORRECT ANSWER: pony tail

<think>The blonde woman’s hair is tied back into a single bunch, which is characteristic of a ponytail.</think>
Answer: pony tail

---

Now given the following image, question, and correct answer:

IMAGE: <image>

QUESTION: {query}
CORRECT ANSWER: {answer}

Please follow the same format as the example above, providing your reasoning and final answer.""",
    "A-OKVQA": 
"""Given an image, a question, and the correct answer, explain step-by-step how the answer can be derived from the image. Please follow the below rules:
- Keep the reasoning concise and grounded in visual or factual evidence. Keep it succinct, within 1-2 sentences.
- Wrap your reasoning in <think> and </think> tags.  
- Then, on the next line, output the final answer, starting with "Answer:".
- The correct answer is provided for reference, but you should derive it from the image and question. Do not refer to the correct answer in your reasoning or answer.

---

Example:

QUESTION: What is the hairstyle of the blond called?
CORRECT ANSWER: pony tail

<think>The man is standing by packed luggage near a road, suggesting he is waiting for a cab or ride.</think>

Answer: cab

---

Now given the following image, question, and correct answer:

IMAGE: <image>

QUESTION: {query}
CORRECT ANSWER: {answer}

Please follow the same format as the example above, providing your reasoning and final answer.""",
"HatefulMemes":
"""Given a meme image and a correct answer ("Yes" if the meme is harmful, "No" if not), write a short reasoning to justify the answer.

- Be faithful to the answer provided.
- First identify the text, answer with "The text on the image is: <text>"
- Focus on the **text**, but also consider how the **image and text work together**.
- If the harm comes only from the text, say so.
- If it comes from the combination, explain that briefly.
- Keep the reasoning inside a single <think>...</think> tag.
- Keep it brief (2–3 sentences).
- On the next line, write the answer as: Answer: Yes or Answer: No.

---

EXAMPLE

CORRECT ANSWER: Yes

<think>The text on the image is: "Love the way you smell today"  
Paired with a skunk image, it sarcastically mocks someone’s body odor.  
This combination is intended to insult and can be harmful.</think>  
Answer: Yes

---

Now complete the following:

IMAGE: <image> 
CORRECT ANSWER: {answer}""",
    "ChartQA": 
"""Given a chart image, a question, and the correct answer, write a short step-by-step reasoning that shows how the answer is derived from the chart.

- Base your reasoning only on the chart data.
- Keep the explanation concise (2–3 sentences max) and factual.
- Wrap your reasoning inside <think> ... </think> tags.
- On the next line, write the final answer in the format: Answer: <answer>

---

EXAMPLE

QUESTION: How many values are below 40 in the Unfavorable graph?
ANSWER: 6

<think>The Unfavorable line is the orange one. From 2005 to 2015, the values below 40 are in years: 2005 (35), 2006 (39), 2007 (39), 2008 (39), 2009 (38), and 2010 (36). That totals 6 values.</think>
Answer: 6

---

Now answer the following:

IMAGE: <image>
QUESTION: {query}
ANSWER: {answer}""",

    "DocVQA":
"""Given a scanned document, a question, and the correct answer, explain step-by-step how the answer can be found in the document.

• Use only information visible in the document.
• Keep your explanation short and factual.
• Wrap the reasoning in <think>...</think> (1–2 sentences max).
• Then output the answer on the next line in the format: Answer: <value>

---

EXAMPLE

QUESTION: Which part of Virginia is this letter sent from?
ANSWER: Richmond

<think>The document header states "Richmond, Virginia," indicating the letter was sent from Richmond.</think>
Answer: Richmond

---

Now complete the following:

IMAGE: <image>
QUESTION: {query}
ANSWER: {answer}""",
    "ImageNet_1K":
"""Given an image, identify the **main object category** it belongs to.
Then explain your reasoning briefly.

- Keep the reasoning short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the predicted category on the next line using: Answer: <answer>
- You are also given the ground-truth category for reference, but do not refer to it in your reasoning.

---

EXAMPLE

GROUND-TRUTH CATEGORY: zucchini, courgette

<think>The image shows a green, elongated vegetable growing on a vine, which appears to be a zucchini. The main object is the zucchini.</think>
Answer: zucchini, courgette

---

Now complete the following:

GROUND-TRUTH CATEGORY: {answer}
IMAGE: <image>
""",

"VOC2007": 
"""Given an image, identify the **main object category** it belongs to.
Then explain your reasoning briefly.

- Keep the reasoning short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the predicted category on the next line using: Answer: <answer>
- You are also given the ground-truth category for reference, but do not refer to it in your reasoning.

---

EXAMPLE

GROUND-TRUTH CATEGORY: zucchini, courgette

<think>The image shows a green, elongated vegetable growing on a vine, which appears to be a zucchini. The main object is the zucchini.</think>
Answer: zucchini, courgette

---

Now complete the following:

GROUND-TRUTH CATEGORY: {answer}
IMAGE: <image>
""",

"Visual7W":
"""Given an image, a question about that image, and the correct answer, explain briefly how the answer can be inferred from the image.

- Use only visual evidence (what can be seen in the image).
- Keep the reasoning short (1–2 sentences max).
- Wrap the reasoning in <think>...</think>.
- Then write the final answer on the next line as: Answer: <answer>

---

EXAMPLE

QUESTION: Where is this taking place?
ANSWER: At a racetrack.

<think>The image shows a person in a suit driving a horse-drawn cart inside a large enclosed arena with dirt ground and spectators behind a barrier—typical features of a racetrack or show ring.</think>
Answer: At a racetrack.

---

Now complete the following:

IMAGE: <image>
QUESTION: {query}
ANSWER: {answer}""",

"SUN397":
"""Given an image, identify the main scene category it depicts. Explain your reasoning briefly.

- Base your reasoning on visual features like objects, activities and setting.
- Keep the explanation short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the final scene label on the next line as: Answer: <scene>
- You are also given the ground-truth scene label for reference, but do not refer to it in your reasoning.

---

EXAMPLE

GROUND-TRUTH SCENE: abbey

<think>The image shows a large stone structure with gothic arches, a tall central tower, and partially ruined walls—features typical of historic monastic buildings. This indicates the scene is an abbey.</think>
Answer: abbey

---

Now answer for the following image with the ground-truth scene label:

GROUND-TRUTH SCENE: {answer}
IMAGE: <image>""",
}




def parse_args():
    parser = argparse.ArgumentParser(description="Generate descriptions using a language model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--subset_name", type=str, nargs='+', required=True, help="Subset names to process.")
    parser.add_argument("--split_name", type=str, nargs='+', default=["test"], help="Split names to process.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--n_partitions", type=int, default=1, help="Number of partitions for the dataset.")
    parser.add_argument("--current_partition", type=int, default=1, help="Current partition index.")
    parser.add_argument("--encode_target", type=str, default="query", help="Encoding query/target.")
    parser.add_argument("--output_folder", type=str, default="descriptions", help="Folder to save descriptions.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use.")

    return parser.parse_args()

def main():

    args = parse_args()

    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,  # Required to load Phi-3.5-vision
        max_model_len=8192,  # Otherwise, it may not fit in smaller GPUs
        limit_mm_per_prompt={"image": 2},  # The maximum number to accept
        tensor_parallel_size=4,
        enforce_eager=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if "internvl" in args.model_name.lower():
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    sampling_params = SamplingParams(max_tokens=512, stop_token_ids=stop_token_ids)

    # if args.prompt_format == 'v2':
    # if args.prompt_format == 'cot':
    #     raise  NotImplementedError("cot prompt format is not supported for GT yet.")

    # else:
    prompts = prompts_reasoning

    n_timeout = 0

    for idx, subset in enumerate(args.subset_name):
        
        prompt = prompts[subset]

        print(f"\033[91m{idx+1}/{len(args.subset_name)}: Processing {subset} now!\033[0m")
        # if args.prompt_format == 'cot':
        #     reasoning_prefix, description_prefix = prefix_keys[subset][args.model_name]
        dataset = load_dataset(args.dataset_name, subset, split=args.split_name[0])

        qry_image_field = "qry_image_path" if "qry_image_path" in dataset.column_names else "qry_img_path"
        qry_text_field = "qry" if "qry" in dataset.column_names else "qry_text"
        tgt_image_field = "tgt_img_path" if "tgt_img_path" in dataset.column_names else "pos_image_path"
        tgt_text_field = "tgt_text" if "tgt_text" in dataset.column_names else "pos_text"

        print([qry_text_field, qry_image_field, tgt_text_field, tgt_image_field])
        print(dataset)


        image_folder = args.image_dir

        folder = os.path.join("descriptions_gt",  subset, "cot")
        os.makedirs(folder, exist_ok=True)
        pkl_files =  [x  for x in os.listdir(folder) if x.endswith(".pkl")]
        descriptions = {}
        if len(pkl_files) > 0:
            print(f"Found existing descriptions in {folder}, loading...")
            for f in pkl_files:
                descriptions.update(pickle.load(open(os.path.join(folder, f), "rb")))

        intermediate_files = [x for x in os.listdir(folder) if x.endswith(".jsonl")]
        if len(intermediate_files) > 0:
            for f in intermediate_files:
                for line in open(os.path.join(folder, f), "r"):
                    line = json.loads(line)
                    descriptions[(line["qry_text"], line["qry_image"])] = line["response"]

        dataset_unprocessed_idx = []

        for idx, row in enumerate(dataset):
            if (row[qry_text_field], row[qry_image_field]) not in descriptions:
                dataset_unprocessed_idx.append(idx)
        
        dataset = dataset.select(dataset_unprocessed_idx)
                
        if args.dataset_split != "test":
            dataset = dataset.to_pandas()
            dataset = dataset.drop_duplicates(subset=[qry_text_field, qry_image_field, tgt_text_field, tgt_image_field])
            dataset = datasets.Dataset.from_pandas(dataset)
        if args.n_partitions > 1:
            dataset = dataset.shard(num_shards=args.n_partitions, index=args.current_partition-1)

        print(dataset)
        print(f"Processing {len(dataset)} images in {subset} for partition {args.current_partition}/{args.n_partitions}")

        intermediates = open(os.path.join(folder, f"intermediates_{args.current_partition}-{args.n_partitions}_{str(os.environ.get('SLURM_JOB_ID'))}.jsonl"), "a") 
        
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), args.batch_size), desc="expand query"):

                batch = dataset[i:i + args.batch_size]


                qry_texts, qry_images, tgt_images, tgt_texts = batch[qry_text_field], batch[qry_image_field], batch[tgt_image_field], batch[tgt_text_field]

                queries = [process_fn(qry, subset) for qry in qry_texts]

                tgt_images = [x[0] for x in tgt_images] if isinstance(tgt_images[0], list) else tgt_images
                tgt_texts = [x[0] for x in tgt_texts] if isinstance(tgt_texts[0], list) else tgt_texts

                if subset in {"MSCOCO"}:
                    loaded_tgt_images = [Image.open(os.path.join(image_folder, tgt_image)) for tgt_image in tgt_images]

                    inputs = [(prompt, tgt_image) for tgt_image in loaded_tgt_images]

                elif subset in {"CIRR"}:

                    qs = [prompt.format(query=q) for q in queries]

                    loaded_qry_images = [Image.open(os.path.join(image_folder, qry_image)) for qry_image in qry_images]
                    loaded_tgt_images = [Image.open(os.path.join(image_folder, tgt_image)) for tgt_image in tgt_images]

                    inputs = [(qry_text, [qry_image, tgt_image]) for qry_text, qry_image, tgt_image in zip(qs, loaded_qry_images, loaded_qry_images)]

                elif subset in {"NIGHTS"}:

                    qs = [prompt for _ in range(len(qry_texts))]

                    loaded_qry_images = [Image.open(os.path.join(image_folder, qry_image)) for qry_image in qry_images]
                    loaded_tgt_images = [Image.open(os.path.join(image_folder, tgt_image)) for tgt_image in tgt_images]

                    inputs = [(qry_text, [qry_image, tgt_image]) for qry_text, qry_image, tgt_image in zip(qs, loaded_qry_images, loaded_qry_images)]

                elif subset in {"VisDial", "MSCOCO_t2i", "VisualNews_t2i", "MSCOCO_i2t"}:

                    qs = [prompt.format(query=q) for q in qry_texts]

                    loaded_tgt_images = [Image.open(os.path.join(image_folder, tgt_image)) for tgt_image in tgt_images]

                    inputs = [(qry_text, [tgt_image]) for qry_text, tgt_image in zip(qs, loaded_tgt_images)]
                
                elif subset in {"WebQA"}:
                    tgt_texts = [x.replace("<|image_1|>\nRepresent the given Wikipedia image with related text information: ", "") for x in tgt_texts]

                    qs = [prompt.format(query=q, target_text=t) for q, t in zip(queries, tgt_texts)]

                    loaded_tgt_images = [Image.open(os.path.join(image_folder, tgt_image)) for tgt_image in tgt_images]

                    inputs = [(qry_text, [tgt_image]) for qry_text, tgt_image in zip(qs, loaded_tgt_images)]
                
                elif subset in {"OK-VQA", "A-OKVQA", "HatefulMemes", "ChartQA", "DocVQA", "ImageNet_1K", "VOC2007", "Visual7W", "SUN397"}:
                    qs = [prompt.format(query=q, answer=t) for q, t in zip(queries, tgt_texts)]
                    loaded_qry_images = [Image.open(os.path.join(image_folder, qry_image)) for qry_image in qry_images]
                    inputs = [(qry_text, [qry_image]) for qry_text, qry_image in zip(qs, loaded_qry_images)]

                formatted_inputs = []

                for qry_text, qry_image in inputs:
                    if qry_image is not None:
                        formatted_inputs.append(
                            {"prompt": tokenizer.apply_chat_template([{"role": "user", "content": qry_text}], add_generation_prompt=True),
                            "multi_modal_data": {"image": qry_image}}
                        )
                    else:
                        formatted_inputs.append(
                            {"prompt": tokenizer.apply_chat_template([{"role": "user", "content": qry_text}], add_generation_prompt=True, tokenize=False)}
                        )
                

                responses = llm.generate(formatted_inputs, sampling_params=sampling_params,)

                for qry, qry_img_path, response in zip(qry_texts, qry_images, responses):
                    descriptions[(qry, qry_img_path)] = response.outputs[0].text

                    intermediates.write(json.dumps({"qry_text": qry, "qry_image": qry_img_path, "response": response.outputs[0].text}) + "\n")
                    intermediates.flush()
                
        print(f"number of timeouts: {n_timeout}")
        pickle.dump(descriptions, open(os.path.join(folder, f"descriptions_{args.dataset_split}_{args.current_partition}-{args.n_partitions}.pkl"), "wb"))
        intermediates.close()
        print_rank(f"Finished processing {subset} for partition {args.current_partition}/{args.n_partitions}.")

if __name__ == "__main__":
    main()
