import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoTokenizer
import torch
from tqdm import tqdm
import pickle
import os
from datasets import load_dataset
import re
import datasets
from vllm import LLM, SamplingParams
from PIL import Image
import argparse

# os.environ["HF_HOME"] = "/opt/dlami/nvme/xuanmingcui/.cache/huggingface"

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']


task_categories = {
    "classification": {"ImageNet-1K", "ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"},
    "vqa": {"OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"},
    "retrieval": {"VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"},
    "grounding": {"MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"}
}

def process_fn_qry(qry, subset):
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

def process_fn_tgt(qry, subset):
    if subset in {"OVEN"}:
        return qry.replace("Represent the given Wikipedia image with related text information: ", "").strip()
    elif subset in {"WebQA"}:
        return qry.replace("<|image_1|>\nRepresent the given Wikipedia image with related text information: ", "").strip()
    elif subset in {"EDIS"}:
        return qry.replace("<|image_1|>\nRepresent the given image with related text information: ", "").strip()
    else:
        return qry


prompt_nogt = {
    "VisDial": 
"""Given a dialogue about an image, write a description of the image in 1-2 sentences.

The caption should:
- Be concise
- Include visual details that are clearly implied or **very likely** given the dialogue
- Be faithful: avoid describing things that contradict or are unsupported by the dialogue
- Use background knowledge where appropriate, but do **not** over-imagine or invent unlikely elements
- Write in a natural, COCO-style caption

Dialogue:
{query}


Caption:
""",
    'captioning': 
"""Given the image below, write a concise and visually detailed description in 2-3 sentences that captures the key elements, objects, people, actions, and scene attributes visible in the image.
- Be concise and detailed while succinct. 
- The description should be factual and grounded in the visible content of the image.
- Do not speculate or infer anything beyond what is clearly shown.
- Focus on what a human viewer would immediately perceive and describe.

Image: <image>

Description:""",
    "VisualNews_i2t": """
You are given a news image. Write a concise, factual, and visually grounded caption that describes what is clearly visible in the image.
- Focus on real-world entities, actions, and settings relevant to news reporting — such as people, places, events, and activities.
- Do not speculate or infer motivations, causes, or unseen context.
- Use clear, objective language to describe the image.
Keep the style formal, informative, and objective — as if writing for a news agency.

Image: <image>

News-style Caption:
""",
    "HatefulMemes":
"""Given the image below, identify and response with the text shown in the image.

Image: <image>

Text:"""
}


prompt_cot_inference = {
    "OK-VQA": 
"""Given an image and a question, explain step-by-step how the answer can be derived from the image. Please follow the below rules:
- Keep the reasoning concise and grounded in visual or factual evidence. Keep it succinct, within 1-2 sentences.
- Wrap your reasoning in <think> and </think> tags.  
- Then, on the next line, output the final answer, starting with "Answer:".

---

Example:

QUESTION: What is the hairstyle of the blond called?

<think>The blonde woman’s hair is tied back into a single bunch, which is characteristic of a ponytail.</think>
Answer: pony tail

---

Now given the following image and question:

IMAGE: <image>

QUESTION: {query}

Please follow the same format as the example above, providing your reasoning and final answer.""",

    "A-OKVQA": 
"""Given an image and a question, explain step-by-step how the answer can be derived from the image. Please follow the below rules:
- Keep the reasoning concise and grounded in visual or factual evidence. Keep it succinct, within 1-2 sentences.
- Wrap your reasoning in <think> and </think> tags.  
- Then, on the next line, output the final answer, starting with "Answer:".

---

Example:

QUESTION: What is the man waiting for?

<think>The man is standing by packed luggage near a road, suggesting he is waiting for a cab or ride.</think>
Answer: cab

---

Now given the following image and question:

IMAGE: <image>

QUESTION: {query}

Please follow the same format as the example above, providing your reasoning and final answer.""",

    "HatefulMemes":
"""Given a meme image containing text, explain step by step if it contains hateful information.

- First identify the text, answer with "The text on the image is: <text>"
- Memes can be naturally sarcastic/humour/exaggerated, while **not** hateful. Therefore, do not treat every sarcasm/humour as hateful.
- Focus on the **text**, but also consider how the **image and text work together**.
- Keep the reasoning inside a single <think>...</think> tag.
- Keep it brief (2–3 sentences).
- On the next line, write the answer as: Answer: Yes or Answer: No.

---

EXAMPLE

<think>The text on the image is: "Love the way you smell today"  
Paired with a skunk image, it sarcastically mocks someone’s body odor.  
This combination is intended to insult and can be harmful.</think>  
Answer: Yes

---

Now answer for the following image:

IMAGE: <image>""",

    "ChartQA": 
"""Given a chart image and a question, write a short step-by-step reasoning that shows how the answer is derived from the chart.

- Base your reasoning only on the chart data.  
- Keep the explanation concise (2–3 sentences max) and factual.  
- Wrap your reasoning inside <think> ... </think> tags.  
- On the next line, write the final answer in the format: Answer: <answer>

---

EXAMPLE

QUESTION: How many values are below 40 in the Unfavorable graph?  

<think>The Unfavorable line is the orange one. From 2005 to 2015, the values below 40 are in years: 2005 (35), 2006 (39), 2007 (39), 2008 (39), 2009 (38), and 2010 (36). That totals 6 values.</think>
Answer: 6

---

Now answer for the following chart image and question:

IMAGE: <image>
QUESTION: {query}""",
    "DocVQA":
"""Given a scanned document and a question, explain step-by-step how the answer can be found in the document.

• Use only information visible in the document.
• Keep your explanation short and factual.
• Wrap the reasoning in <think>...</think> (1–2 sentences max).
• Then output the answer on the next line in the format: Answer: <value>

---

EXAMPLE

QUESTION: Which part of Virginia is this letter sent from?

<think>The document header states "Richmond, Virginia," indicating the letter was sent from Richmond.</think>
Answer: Richmond

---

Now answer for the following image and question:

IMAGE: <image>
QUESTION: {query}""",

    "VOC2007":
"""Given an image, identify the **main object category** it belongs to.
Then explain your reasoning briefly.

- Use visual cues such as shape, color, and context.
- Keep the reasoning short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the predicted category on the next line using: Answer: <answer>

---

EXAMPLE

<think>The image mainly shows a green, elongated vegetable growing on a vine, consistent with a zucchini.</think>
Answer: zucchini, courgette

---

Now answer for the following image:

IMAGE: <image>""",

    "ImageNet_1K":
"""Given an image, identify the **main object category** it belongs to.
Then explain your reasoning briefly.

- Use visual cues such as shape, color, and context.
- Keep the reasoning short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the predicted category on the next line using: Answer: <answer>

---

EXAMPLE

<think>The image mainly shows a green, elongated vegetable growing on a vine, consistent with a zucchini.</think>
Answer: zucchini, courgette

---

Now answer for the following image:

IMAGE: <image>""",

    "ImageNet-1K":
"""Given an image, identify the **main object category** it belongs to.
Then explain your reasoning briefly.

- Use visual cues such as shape, color, and context.
- Keep the reasoning short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the predicted category on the next line using: Answer: <answer>

---

EXAMPLE

<think>The image mainly shows a green, elongated vegetable growing on a vine, consistent with a zucchini.</think>
Answer: zucchini, courgette

---

Now answer for the following image:

IMAGE: <image>""",


"Visual7W":
"""Given an image, and a question about that image, explain briefly how the answer can be inferred from the image.

- Use only visual evidence (what can be seen in the image).
- Keep the reasoning short (1–2 sentences max).
- Wrap the reasoning in <think>...</think>.
- Then write the final answer on the next line as: Answer: <answer>

---

EXAMPLE

QUESTION: Where is this taking place?

<think>The image shows a person in a suit driving a horse-drawn cart inside a large enclosed arena with dirt ground and spectators behind a barrier—typical features of a racetrack or show ring.</think>
Answer: At a racetrack.

---

Now answer for the following image and question:

IMAGE: <image>
QUESTION: {query}""",


"N24News": 
"""Given a news image and its caption, determine the **main news domain** it belongs to.

- Use only information visible in the image and the caption.
- Wrap your reasoning in <think>...</think> (1–2 sentences).
- Output the domain on the next line in the format: Answer: <domain>

---

EXAMPLE

<think>The image shows baseball players celebrating, and the caption describes a Game 7 win, which is a sports context.</think>
Answer: Sports

---

Now answer for the following image:

IMAGE: <image>
CAPTION: {query}""",
    "InfographicsVQA": 
"""Given an infographic image and a question, explain how the answer is derived from the infographic.

- Use only information visible in the infographic.
- Keep your explanation brief (1–2 sentences).
- Wrap your reasoning inside <think>...</think>.
- Then output the answer on the next line in the format: Answer: <answer>

---

EXAMPLE

QUESTION: Which type of fonts offer better readability in printed works?

<think>The infographic states that serif fonts are easier to read in printed works because their stroke details help guide the horizontal flow of the eyes.</think>
Answer: serif fonts

---

Now answer for the following image:

IMAGE: <image>
QUESTION: {query}""",
"MSCOCO_i2t":
"""Given an image, describe briefly how a COCO-style caption can be formed from the image.

- First provide a detailed but succinct description of the image. 
- Then describe what a COCO-style caption should contain. (Hint: it should focus on the most salient objects and their arrangement in the image.)
• Wrap your reasoning in <think>...</think> (2–3 sentences).  
• Then write the final COCO-style caption on the next line as: Answer: <answer>

---

EXAMPLE

<think>The image shows a cozy bedroom with a wooden bed, striped bedsheets, a lamp on the nightstand with its light turned on, and several large pillows arranged neatly from head to foot along the bed.
The COCO-style caption should contain the most salient object and arrangement: the pillows.</think>  
Answer: Several pillows are lined up down the length of a bed.

---

Now answer for the following image:

IMAGE: <image>""",
"VisualNews_i2t":
"""Given an image, describe briefly how the news caption can be derived from the image.
- Start by visually describing key elements in the image (e.g., actions, people, location clues, uniforms, text).
- Then use your background knowledge to reason about what news event or context is likely being captured.
- Identify known public figures or logos if recognizable.
- Keep your reasoning concise (2–3 sentences).
- Wrap your reasoning in <think>...</think>.
- Then output the final caption on the next line using: Answer: <answer>

---

EXAMPLE

<think>The image shows two British athletes in GBR uniforms, one helping the other as they near a finish line. A banner with “México” and cheering crowds suggest an international sports event. Based on the uniforms and gesture, this likely depicts Alistair Brownlee aiding his brother Jonny during a triathlon in Mexico.</think>
Answer: British athlete Alistair Brownlee helps his brother Jonny Brownlee before crossing the finish line during the 2016 ITU World Triathlon Grand Final in Cozumel, Mexico.

---

Now answer for the following image:

IMAGE: <image>""",

"SUN397":
"""Given an image, identify the main scene category it depicts. Explain your reasoning briefly.

- Base your reasoning on visual features like objects, activities and setting.
- Keep the explanation short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the final scene label on the next line as: Answer: <scene>

---

EXAMPLE

<think>The image shows a large stone structure with gothic arches, a tall central tower, and partially ruined walls—features typical of historic monastic buildings. This indicates the scene is an abbey.</think>
Answer: abbey

---

Now answer for the following image:

IMAGE: <image>""",

"CIRR": """Given a base image and a textual modification instruction,  
reason about what the target image should likely look like **after applying the described change**.

NOTES:
• The target image may differ significantly from the base.  
• Only the parts explicitly stated in the instruction should carry over.  
• Do NOT assume background, pose, or angle remain the same unless stated.  
• Wrap your reasoning in <think>...</think> (1–3 sentences).
• Then write a caption of the likely target image on the next line using: Answer: <answer>
- The caption should solely about the target image. Do not mention or compare with the base image and the modification instruction.

---

EXAMPLE

INSTRUCTION: Shows the front of a younger lighter colored beaver standing on a large rock with dry leaves in the background.

<think> The base image shows a dark-colored beaver-like animal in side view on a rock.  
The instruction asks for a younger, lighter-colored beaver, seen from the front, with dry leaves behind.  
Since backgrounds may differ, only those mentioned should be included — in this case, the animal’s appearance, pose, and leaf-covered setting. </think>  
Answer: A lighter-colored young beaver stands facing forward on a large rock, with dry leaves visible in the background.

---

Now complete the following:

BASE IMAGE: <image>
INSTRUCTION: {query}""",

"FashionIQ": """Given a base image of a clothing item and a modification instruction,  
describe what the target garment should look like **after applying the instruction**.

IMPORTANT:
• Focus ONLY on the clothing (not the person, pose, or background).
• Wrap your reasoning in <think>...</think> (1–3 sentences).
• Then write a short description of the likely target garment using: Answer: <description>

---

EXAMPLE

INSTRUCTION: Has thin straps and different pattern and more autumn colored and longer

<think>The base dress is strapless with bold horizontal stripes in bright colors.  
The instruction asks for thin shoulder straps, a different pattern, more fall-like colors, and added length.  
So the target should be a long dress with thin straps, an intricate or floral pattern, and warm autumn tones.</think>  
Answer: A long dress with thin straps, featuring a multi-patterned design in warm, autumnal colors and a flowing shape.

---

Now complete for the following:

BASE IMAGE: <image>
INSTRUCTION: {query}""",

"Place365":
"""Given an image, identify the main scene category it depicts. Explain your reasoning briefly.

- Base your reasoning on visual features like objects, activities and setting.
- Keep the explanation short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the final scene label on the next line as: Answer: <scene>

---

EXAMPLE

<think>The image shows sunlight streaming from behind a large cloud against a clear blue background. The absence of ground or structures indicates the scene is the sky.</think>  
Answer: sky

---

Now answer for the following image:

IMAGE: <image>""",

    "ImageNet-A":
"""Given an image, identify the **main object category** it belongs to.
Then explain your reasoning briefly.

- Use visual cues such as shape, color, and context.
- Keep the reasoning short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the predicted category on the next line using: Answer: <answer>

---

EXAMPLE

<think>The image mainly shows a green, elongated vegetable growing on a vine, consistent with a zucchini.</think>
Answer: zucchini, courgette

---

Now answer for the following image:

IMAGE: <image>""",

    "ImageNet-R":
"""Given an image, identify the **main object category** it belongs to.
Then explain your reasoning briefly.

- Use visual cues such as shape, color, and context.
- Keep the reasoning short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the predicted category on the next line using: Answer: <answer>

---

EXAMPLE

<think>The image mainly shows a green, elongated vegetable growing on a vine, consistent with a zucchini.</think>
Answer: zucchini, courgette

---

Now answer for the following image:

IMAGE: <image>""",

    "ObjectNet":
"""Given an image, identify the object category of the main object in the image.
Then explain your reasoning briefly.

- Use visual cues such as shape, color, and context.
- Keep the reasoning short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the predicted category on the next line using: Answer: <answer>

---

EXAMPLE

<think>The image mainly shows a green, elongated vegetable growing on a vine, consistent with a zucchini.</think>
Answer: zucchini, courgette

---

Now answer for the following image:

IMAGE: <image>""",

    "Country211":
"""Given an image, identify the country where the image is taken place. Explain your reasoning briefly.

- Use visual cues such as objects and context.
- Keep the reasoning short (1–2 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the predicted country on the next line using: Answer: <answer>

---

EXAMPLE

<think>The image shows a buffet with Chinese dishes like braised meats, tofu, roasted duck, and red chili peppers, which are typical of Chinese cuisine.</think>
Answer: Chinese

---

Now answer for the following image:

IMAGE: <image>""",

"TextVQA": """Given an image that may contain text and a question about it
- think briefly about where the required text or clue appears
- read or infer the answer
- wrap your reasoning in <think> … </think> (1-3 sentences)
- then write the answer on the next line in the format: Answer: <answer>

---

EXAMPLE

QUESTION: What does the small white text spell?

<think>The banner reads "DRUPALCON" in large letters, and just beneath it the smaller white text spells "COPENHAGEN".</think>
Answer: copenhagen

---

Now complete the following:

IMAGE: <image> 
QUESTION: {query}""",
    "GQA": 
"""Given an image and a question, explain step-by-step how the answer can be derived from the image. Please follow the below rules:
- First describe the image briefly, focusing on key elements.
- Keep the reasoning concise and grounded in visual or factual evidence. Keep it succinct, within 1-2 sentences.
- Wrap your reasoning in <think> and </think> tags.  
- Then, on the next line, output the final answer, starting with "Answer:".

---

Example:

QUESTION: Does the man ride a horse?

<think>The image shows a street scene with parked cars on both sides and a person riding in a bike lane. The person is clearly on a bicycle, with two wheels, handlebars, and pedals visible. There is no horse in the scene.</think>
Answer: no, the man rides a bike

---

Now given the following image and question:

IMAGE: <image>

QUESTION: {query}

Please follow the same format as the example above, providing your reasoning and final answer.""",

    "VizWiz": 
"""Given an image and a question, explain step-by-step how the answer can be derived from the image. Please follow the below rules:
- First describe the image briefly, focusing on key elements.
- Keep the reasoning concise and grounded in visual or factual evidence. Keep it succinct, within 1-2 sentences.
- Wrap your reasoning in <think> and </think> tags.  
- Then, on the next line, output the final answer, starting with "Answer:".

---

Example:

QUESTION: Which one of these items is the children's dictionary? Is it the one on the right, or the one on the left?

<think>The item on the left has bright green packaging labeled "Children’s Dictionary" with a smiling child on it. The one on the right says "Spell Corrector & Puzzle Solver" and has a more neutral design.</think>
Answer: left

---

Now given the following image and question:

IMAGE: <image>

QUESTION: {query}

Please follow the same format as the example above, providing your reasoning and final answer.""",

    "ScienceQA": 
"""Given a science-related image and a question, explain step-by-step how the answer can be derived from the image. Please follow the below rules:
- First describe the image briefly, focusing on the scientific elements or phenomena depicted.
- Keep the reasoning concise and grounded in scientific or factual evidence. Keep it succinct, within 1-2 sentences.
- Wrap your reasoning in <think> and </think> tags.  
- Then, on the next line, output the final answer, starting with "Answer:".

---

Example:

QUESTION: What is the name of the colony shown?

<think>The image is a map of the original thirteen American colonies. One of the colonies is highlighted in dark green. This colony is located in the northeastern region, between Massachusetts and Maine, which corresponds to the modern state of New Hampshire.</think>
Answer: Hampshire

---

Now given the following image and question:

IMAGE: <image>

QUESTION: {query}

Please follow the same format as the example above, providing your reasoning and final answer.""",

    "Wiki-SS-NQ": 
"""Given a question, explain what information is needed to answer the question. Please follow the below rules:
- Keep the reasoning concise and grounded in scientific or factual evidence. Keep it succinct, within 1-2 sentences.
- Wrap your reasoning in <think> and </think> tags.  
- Then, on the next line, output the final summarized answer of what information is needed to support the question, starting with "Answer:".

---

Example:

QUESTION: who needs to know about the jet stream?

<think>The correct document should discuss practical uses of jet stream knowledge. This includes how pilots use it to optimize flight routes and how meteorologists rely on it for weather forecasting.</think>
Answer: The document should discuss users of jet stream.

---

Now given the following image and question:

IMAGE: <image>

QUESTION: {query}

Please follow the same format as the example above, providing your reasoning and final answer.""",

"OVEN": """Given a query image and a question, the overall task is to retrieve the target relevant image-text pair that provides information about the question.
Now, your task is to explain step-by-step what information should the target image-text pair contains, and then provide an answer that summarizes the information needed.

Query question: Where is this building?

<think>The query image shows a snow-covered lakeside with a wooden pier, surrounded by a forested and mountainous area. To retrieve the correct target, we need an image-text pair that features the same lake or geographic area. The image should show a similar natural body of water surrounded by hills or mountains. The associated text should mention the name of the lake and provide geographic context that can confirm its identity.</think>
Answer: Looking for a pair where the image shows the same lake from another angle and the text confirms its name and location.

Please generate your response following the same format as above, using the query image and question below:

IMAGE: <image>

QUESTION: {query}
""",

    "VisualNews_t2i":
"""Given a news caption, use your background knowledge and reasoning to expand the caption with more visual details for the corresponding image. Follow below rules:

- Generate succinct reasoning process (2-3 sentences) about the visual information that can be inferred from the caption, using your background knowledge.
- Wrap your reasoning in <think>...</think>.
- Then output the final description about the corresponding image on the next line using: Answer: <answer>

---

EXAMPLE

CAPTION: Turkish President Tayyip Erdogan looks on after arriving at Esenboga Airport faced the prospect of weeks of political turmoil after the ruling AK Party lost its parliamentary majority

<think>The caption describes a political moment following an election loss, so Erdogan is likely shown with a serious or solemn expression. As a head of state arriving at an airport, he would be dressed formally and surrounded by security or aides in suits. The setting is likely outdoors, near an official vehicle or terminal.</think>
Answer: Turkish President Tayyip Erdogan walks with a solemn expression, wearing a formal suit, closely accompanied by suited security personnel in an outdoor airport setting.

---

Now answer for the following caption:

CAPTION: {query}""",

"MSCOCO_t2i": """Given a caption describing an everyday scene, use your background knowledge and reasoning to expand the caption with more concrete and visual details. Follow the rules below:

- Generate a succinct reasoning process (2–3 sentences) about the visual information that can be inferred from the caption, using commonsense and visual priors.
- Wrap your reasoning in <think>...</think>.
- Then output the final description of the corresponding image on the next line using: Answer: <answer>

---

EXAMPLE

CAPTION: A teddy bear shop is equipped with a door guard teddy and a neighbor teddy above.

<think>The caption playfully describes a teddy bear shop, suggesting there are teddy bear figures used decoratively. A “door guard teddy” likely refers to a large bear figure placed at the shop entrance, possibly dressed in a costume to resemble a guard. The “neighbor teddy above” suggests another teddy is positioned at or hanging from an upper-story window, adding to the shop’s whimsical appearance.</think>
Answer: A teddy bear shop with a large teddy bear dressed as a royal guard standing at the entrance, and another teddy bear playfully hanging from an upstairs window.

---

Now answer for the following caption:

CAPTION: {query}""",

"NIGHTS": """Given a query image, use your background knowledge and visual reasoning to describe what kinds of visual features should be present in a matching target image. Your goal is to help a model retrieve a visually and contextually similar image based on salient elements. Follow the rules below:

- Generate a succinct reasoning process (2–3 sentences) that highlights the key visual or contextual features in the query image. This may include scene type, objects, structure, layout, lighting, textures, or mood.
- Wrap your reasoning in <think>...</think>.
- Then output the final description of the query image on the next line using: Answer: <answer>

---

EXAMPLE

<think>The query image shows an outdoor restaurant scene at dusk, with modern architecture, visible indoor lighting, and set tables on a clean patio. To match it, the target image should depict a similar setting: an outdoor or semi-outdoor dining area, with aligned rows of tables, modern decor, prominent artificial lighting, and a warm or evening ambiance. The structure should feel open, inviting, and designed for dinner service.</think>
Answer: A modern outdoor dining area at dusk, featuring neatly arranged tables with white cloths, warm ambient lighting, and a stylish restaurant interior visible through large openings.

---

Now answer for the following query image:

IMAGE: <image>""",

"WebQA": """Given a question, use your reasoning to determine what kind of image-text pair would help answer the question. Consider what visual or textual information is needed to provide the answer. Follow the rules below:

- Generate a succinct reasoning process (2–3 sentences) explaining what content should be present in the image and/or text to answer the question.
- Wrap your reasoning in <think>...</think>.
- Then output a summary of what the correct target image-text pair should contain using: Answer: <answer>

---

EXAMPLE

QUESTION: Is more of the building on the corner of King William St–Gracechurch St green or grey?  
TARGET TEXT: King William St–Gracechurch St

<think>To answer the question about the building's color at the corner of King William Street and Gracechurch Street, the target image-text pair should clearly identify the intersection and show the relevant building from a view that reveals its dominant exterior colors. The image should feature a large portion of the building’s facade and rooftop, while the text should confirm the location for grounding.</think>  
Answer: A photo of the intersection at King William Street and Gracechurch Street, showing a corner building with mostly grey stone exterior and a green rooftop, accompanied by text confirming the street names.

---

Now generate your response for the following question :
QUESTION: {query}""",

"EDIS": """Given a news caption, use your background knowledge and reasoning to expand the caption with more visual and textual cues that would help retrieve a matching image-text pair. Follow the rules below:

- Generate a succinct reasoning process (2–3 sentences) about the visual and textual information that can be inferred or expected from the caption.
- Wrap your reasoning in <think>...</think>.
- Then output the final answer as a description of what the target image-text pair should contain using: Answer: <answer>

---

EXAMPLE

CAPTION: Former Israeli president and prime minister Shimon Peres right and Palestinian President Mahmoud Abbas arrive at the World Economic Forum in Southern Shuneh Jordan in May 2015.  
TARGET TEXT: World — As the West pays tribute to Peres, many Arabs recall a legacy of destruction.

<think>The caption names two political figures—Shimon Peres and Mahmoud Abbas—attending an international forum. Visually, the image likely includes both men in suits, in a formal indoor setting, walking or posing for the press. For retrieval, the matching image-text pair should visually show the two leaders together and textually provide geopolitical or historical framing, particularly referencing Peres's controversial legacy and broader Middle Eastern politics.</think>
Answer: Shimon Peres and Mahmoud Abbas in formal attire at an indoor event, with text highlighting differing global reactions to Peres’s legacy, including Arab perspectives.

---

Now answer for the following caption:

CAPTION: {query}
""",

"RefCOCO": """Given an image and a referring expression, your task is to first reason about what object or region the expression refers to, then generate a detailed description of that object or region. Follow the rules below:

- Generate a brief reasoning process (1–2 sentences) that identifies what and where the expression refers to.
- Wrap your reasoning in <think>...</think>.
- Then output a detailed description of the referred object or region using: Answer: <answer>
- Only describe the referred object/region and the background directly behind it. Do not describe any other parts of the image.

---

EXAMPLE

QUERY EXPRESSION: bowl behind the others can only see part  

<think>The expression refers to the partially visible bowl in the back-right of the image, mostly hidden behind other dishes. It appears to be a white ceramic bowl with vertical ridges and a light blue band near its base.</think>  
Answer: A white, rib-sided ceramic bowl with a soft blue band near its base, only its upper rim and curved side visible; it rests on a light-colored cloth atop the table, with muted countertop surface behind.

---

Now answer for the following referring expression:  
QUERY EXPRESSION: {query}
QUERY IMAGE: <image>""",

"MSCOCO": """Given an image and a referring expression, your task is to first reason about what object or region the expression refers to, then generate a detailed description of that object or region. Follow the rules below:

- Generate a brief reasoning process (1–2 sentences) that identifies what and where the expression refers to.
- Wrap your reasoning in <think>...</think>.
- Then output a detailed description of the referred object or region using: Answer: <answer>
- Only describe the referred object/region and the background directly behind it. Do not describe any other parts of the image.

---

EXAMPLE

QUERY EXPRESSION: bowl behind the others can only see part  

<think>The expression refers to the partially visible bowl in the back-right of the image, mostly hidden behind other dishes. It appears to be a white ceramic bowl with vertical ridges and a light blue band near its base.</think>  
Answer: A white, rib-sided ceramic bowl with a soft blue band near its base, only its upper rim and curved side visible; it rests on a light-colored cloth atop the table, with muted countertop surface behind.

---

Now answer for the following referring expression:  
QUERY EXPRESSION: {query}
QUERY IMAGE: <image>""",

"Visual7W-Pointing": """Given an image and a referring expression, your task is to first reason about what object or region the expression refers to, then generate a detailed description of that object or region. Follow the rules below:

- Generate a brief reasoning process (1–2 sentences) that identifies what and where the expression refers to.
- Wrap your reasoning in <think>...</think>.
- Then output a detailed description of the referred object or region using: Answer: <answer>
- Only describe the referred object/region and the background directly behind it. Do not describe any other parts of the image.

---

EXAMPLE

QUERY EXPRESSION: bowl behind the others can only see part  

<think>The expression refers to the partially visible bowl in the back-right of the image, mostly hidden behind other dishes. It appears to be a white ceramic bowl with vertical ridges and a light blue band near its base.</think>  
Answer: A white, rib-sided ceramic bowl with a soft blue band near its base, only its upper rim and curved side visible; it rests on a light-colored cloth atop the table, with muted countertop surface behind.

---

Now answer for the following referring expression:  
QUERY EXPRESSION: {query}
QUERY IMAGE: <image>""",

"RefCOCO-Matching": """Given an image and a referring expression, your task is to first reason about what object or region the expression refers to, then generate a detailed description of that object or region. Follow the rules below:

- Generate a brief reasoning process (1–2 sentences) that identifies what and where the expression refers to.
- Wrap your reasoning in <think>...</think>.
- Then output a detailed description of the referred object or region using: Answer: <answer>
- Only describe the referred object/region and the background directly behind it. Do not describe any other parts of the image.

---

EXAMPLE

QUERY EXPRESSION: bowl behind the others can only see part  

<think>The expression refers to the partially visible bowl in the back-right of the image, mostly hidden behind other dishes. It appears to be a white ceramic bowl with vertical ridges and a light blue band near its base.</think>  
Answer: A white, rib-sided ceramic bowl with a soft blue band near its base, only its upper rim and curved side visible; it rests on a light-colored cloth atop the table, with muted countertop surface behind.

---

Now answer for the following referring expression:  
QUERY EXPRESSION: {query}
QUERY IMAGE: <image>""",

        "VisDial": """Given a dialogue about a target image, analyze the dialogue and generate a concise and detailed caption of the target image. Explain your reasoning briefly. Follow the below rules:

- Keep the reasoning short (1–3 sentences).
- Wrap the reasoning in <think>...</think>.
- Output the predicted category on the next line using: Answer: <answer>

---

Example:

DIALOGUE: Q:is the picture in color
A:yes
Q:how old does the woman look
A:maybe late 20's to early 30's
Q:are there any other people
A:no, just her and the bird
Q:are there any animals
A:just the bird
Q:what is she feeding the bird
A:i can't tell, but it looks like a small rodent
Q:is she wearing a hat
A:yes
Q:what color is her shirt
A:mauve
Q:what color is her hat
A:blonde
Q:what kind of bird is it
A:a very large black bird with a white ring around his neck
Q:what else do you see
A:it appears to be at a zoo there are trees behind her

<think>The dialogue indicates a zoo-like setting where a woman in a mauve shirt and hat is feeding a large black bird with a white ring around its neck. She appears to be alone with the bird, and the item being fed resembles a small rodent. Trees are visible in the background.</think>
Answer: A woman in a mauve shirt and hat is feeding a large black bird with a white ring around its neck at what appears to be a zoo.

Now answer for the following dialogue:
DIALOGUE: {query}""",

}

prompt_cot_inference_target = {
    "NIGHTS": """Given a query image, use your background knowledge and visual reasoning to describe what kinds of visual features should be present in a matching target image. Your goal is to help a model retrieve a visually and contextually similar image based on salient elements. Follow the rules below:

- Generate a succinct reasoning process (2–3 sentences) that highlights the key visual or contextual features in the query image. This may include scene type, objects, structure, layout, lighting, textures, or mood.
- Wrap your reasoning in <think>...</think>.
- Then output the final description of the query image on the next line using: Answer: <answer>

---

EXAMPLE

<think>The query image shows an outdoor restaurant scene at dusk, with modern architecture, visible indoor lighting, and set tables on a clean patio. To match it, the target image should depict a similar setting: an outdoor or semi-outdoor dining area, with aligned rows of tables, modern decor, prominent artificial lighting, and a warm or evening ambiance. The structure should feel open, inviting, and designed for dinner service.</think>
Answer: A modern outdoor dining area at dusk, featuring neatly arranged tables with white cloths, warm ambient lighting, and a stylish restaurant interior visible through large openings.

---

Now answer for the following query image:

IMAGE: <image>""",

    "VisDial": """Given an image, generate a detailed, concise caption that can match with a target dialogue that discusses about the image.

- Generate a succinct reasoning process (2–3 sentences) that highlights the key visual or contextual features in the image.
- Wrap your reasoning in <think>...</think>.
- Then output the final description of the query image on the next line using: Answer: <answer>

---

EXAMPLE

<think>The query image shows an outdoor restaurant scene at dusk, with modern architecture, visible indoor lighting, and set tables on a clean patio. To match it, the target dialogue should depict a similar setting: an outdoor or semi-outdoor dining area, with aligned rows of tables, modern decor, prominent artificial lighting, and a warm or evening ambiance. The structure should feel open, inviting, and designed for dinner service.</think>
Answer: A modern outdoor dining area at dusk, featuring neatly arranged tables with white cloths, warm ambient lighting, and a stylish restaurant interior visible through large openings.

---

Now answer for the following query image:

IMAGE: <image>""",

"VisualNews_t2i": prompt_cot_inference['VisualNews_i2t'],
"VisualNews_i2t": prompt_cot_inference['VisualNews_t2i'],
"MSCOCO_t2i": prompt_cot_inference['MSCOCO_i2t'],
"MSCOCO_i2t": prompt_cot_inference['MSCOCO_t2i'],

    "CIRR": """Given a query image, first describe the image in detail, then produce a COCO-style succinct caption. Follow the rules below:

- Generate a reasoning process (2–3 sentences) that describes the image in fine-grained details.
- Wrap your reasoning in <think>...</think>.
- Then output a COCO-style caption using: Answer: <answer>

---

EXAMPLE

<think>The image shows a large brown dog, likely a Bloodhound, sitting on green grass in an outdoor park or showground. The dog has long, floppy ears, loose skin, and a prominent snout, which are characteristic of the breed. Its posture is upright but relaxed, and it's looking slightly to the side. Behind the dog, there are metal barricades and signs, as well as trees and tents, featuring a public outdoor setting.</think>
Answer: A brown Bloodhound sitting on grass in front of meta fence.
---

Now answer for the following query image:

IMAGE: <image>""",

    "WebQA": """Given an wiki image and a short text description, first describe the image in detail incorporating relevant information from the accompanying text, then produce a short summary. Follow the rules below:

- Generate a reasoning process (2–3 sentences) that describes the image in fine-grained details including the textual information.
- Wrap your reasoning in <think>...</think>.
- Then output a COCO-style caption using: Answer: <answer>

---

EXAMPLE

<think>The image shows Holy Family Church in Orange, California, which previously held cathedral status. It is a red-brick church with a sharply peaked roof and a tall, narrow bell tower on the left topped with a cross. The building has a simple, symmetrical design with dark trim, small entrance columns, and a religious mural or emblem above the main doors. Surrounding the church are a few trees and a small lawn, and the overcast sky gives the scene a muted atmosphere.</think>
Answer: The image shows, Holy Family Church, a red-brick church with a tall bell tower located in Orange, California.

---

Now answer for the following image and text:

TEXT: {query}
IMAGE: <image>""",

    "FashionIQ": """Given a fashion image, first describe the garment in detail, then produce a short description of the garment. Follow the rules below:

- Generate a reasoning process (2–3 sentences) that describes the garment in fine-grained details.
- Focus solely on the clothing item, ignoring the background and person wearing it.
- Wrap your reasoning in <think>...</think>.
- Then output a COCO-style caption using: Answer: <answer>

---

EXAMPLE

<think>The garment is a sleeveless, full-length maxi dress with thin spaghetti straps and a V-neckline. It features a patchwork-style design composed of multiple horizontal bands of colorful patterns, including florals, abstract prints, and geometric motifs. The vivid palette leans toward earthy and autumn tones with pops of red, pink, green, blue, and cream, giving the dress a bohemian and eclectic feel.</think>
Answer: A bohemian patchwork maxi dress with thin straps and vibrant multicolored horizontal patterns.

---

Now answer for the following image and text:

IMAGE: <image>""",

    "MSCOCO": """Given an image, first describe the image in detail, then produce a COCO-style succinct caption. Follow the rules below:

- Generate a reasoning process (2–3 sentences) that describes the image in fine-grained details.
- Wrap your reasoning in <think>...</think>.
- Then output a COCO-style caption using: Answer: <answer>

---

EXAMPLE

<think>The image shows a large brown dog, likely a Bloodhound, sitting on green grass in an outdoor park or showground. The dog has long, floppy ears, loose skin, and a prominent snout, which are characteristic of the breed. Its posture is upright but relaxed, and it's looking slightly to the side. Behind the dog, there are metal barricades and signs, as well as trees and tents, featuring a public outdoor setting.</think>
Answer: A brown Bloodhound sitting on grass in front of meta fence.

---

Now answer for the following query image:

IMAGE: <image>""",

    "Wiki-SS-NQ": """Given a document screenshot, first describe the document in detail, then produce a short summary of the document. Follow the rules below:

- Generate a reasoning process (2–4 sentences) that describes the document in fine-grained details.
- Wrap your reasoning in <think>...</think>.
- Then output a short summary using: Answer: <answer>

---

EXAMPLE

<think>The document is a Wikipedia page about Destiny USA, a large super-regional shopping and entertainment complex in Syracuse, New York. It highlights key facts such as Destiny USA being the largest mall in New York State and among the most visited in the U.S. The page describes its layout across six aboveground and one underground floor, housing retail shops, food courts, cinemas, and parking. The mall opened in 1990 as Carousel Center and underwent significant expansions and renovations. Visual elements on the right include the mall’s logo, an aerial photograph, and a map pinpointing its location. Additional sections detail the mall’s history, background, and construction origins on a former scrapyard.</think>
Answer: A Wikipedia article detailing the history, structure, and significance of Destiny USA, a major shopping mall in Syracuse, New York, with visuals including its logo, location map, and aerial view.

---

Now answer for the following query image:

IMAGE: <image>""",

    "RefCOCO": """Given an image, first describe the image in detail, then produce a COCO-style succinct caption. Follow the rules below:

- Generate a reasoning process (2–3 sentences) that describes the image in fine-grained details.
- Wrap your reasoning in <think>...</think>.
- Then output a COCO-style caption using: Answer: <answer>

---

EXAMPLE

<think>The image shows a large brown dog, likely a Bloodhound, sitting on green grass in an outdoor park or showground. The dog has long, floppy ears, loose skin, and a prominent snout, which are characteristic of the breed. Its posture is upright but relaxed, and it's looking slightly to the side. Behind the dog, there are metal barricades and signs, as well as trees and tents, featuring a public outdoor setting.</think>
Answer: A brown Bloodhound sitting on grass in front of meta fence.

---

Now answer for the following query image:

IMAGE: <image>""",

    "Visual7W-Pointing": """Given an image, first describe the image in detail, then produce a COCO-style succinct caption. Follow the rules below:

- Generate a reasoning process (2–3 sentences) that describes the image in fine-grained details.
- Wrap your reasoning in <think>...</think>.
- Then output a COCO-style caption using: Answer: <answer>

---

EXAMPLE

<think>The image shows a large brown dog, likely a Bloodhound, sitting on green grass in an outdoor park or showground. The dog has long, floppy ears, loose skin, and a prominent snout, which are characteristic of the breed. Its posture is upright but relaxed, and it's looking slightly to the side. Behind the dog, there are metal barricades and signs, as well as trees and tents, featuring a public outdoor setting.</think>
Answer: A brown Bloodhound sitting on grass in front of meta fence.

---

Now answer for the following query image:

IMAGE: <image>""",

    "OVEN": """Given an wiki image and a text description, first describe the image in detail incorporating relevant information from the accompanying text, then produce a short summary. Follow the rules below:

- Generate a reasoning process (2–3 sentences) that describes the image in fine-grained details including the textual information.
- Wrap your reasoning in <think>...</think>.
- Then output a COCO-style caption using: Answer: <answer>

---

EXAMPLE

TEXT: Titisee. The Titisee is a lake in the southern Black Forest in Baden-Württemberg. It covers an area of 1.3 (km2) and is an average of 20 (m) deep. It owes its formation to the Feldberg glacier, the moraines of which were formed in the Pleistocene epoch and nowadays form the shores of the lake. The lake's outflow, at 840 (m) above sea level, is the River Gutach, which merges with the Haslach stream below Kappel to form the Wutach. The waters of the Titisee thus drain eventually into the Upper Rhine between Tiengen and Waldshut. On the north shore lies the.

<think>The image depicts a serene mountain lake surrounded by densely packed, dark green coniferous forest. The lake has a calm, reflective surface and is framed by gently sloping hills and higher ridgelines in the background, giving a sense of elevation and enclosure. On the far side of the lake, a small town is visible, with clusters of red-roofed buildings and patches of open grassland stretching along the shoreline. The landscape is layered, with mist or light haze softening the distant hills. This visually matches the description of the Titisee, a glacial lake in the Black Forest region of Baden-Württemberg, Germany, which formed during the Pleistocene epoch and lies at 840 meters above sea level.</think>
Answer: A calm mountain lake surrounded by forested hills and a town along the shore, visually representing the Titisee in Germany's Black Forest.

---

Now answer for the following image and text:

TEXT: {query}
IMAGE: <image>""",

"EDIS": """Given a news image and an accompanying text about an event, first describe the image in detail and reason about how it symbolically or emotionally connects to the event described in the text. Then produce a short summary. Follow the rules below:

- Generate a reasoning process (2–3 sentences) that describes the visual elements in fine-grained detail and connects them meaningfully to the text.
- Wrap your reasoning in <think>...</think>.
- Then output a COCO-style caption using: Answer: <answer>

---

EXAMPLE

TEXT: After Newtown shooting, mourning parents enter into the lonely quiet

<think>The image shows a man holding a large framed photo of three smiling children on a beach, their expressions full of life and playfulness. Beneath the framed photo lie other smaller prints on a table, suggesting a personal collection of cherished memories. Although the image does not show the tragedy itself, it symbolically conveys the grief and remembrance of parents who lost children in the Newtown shooting—preserving their joy through photos while facing an overwhelming absence.</think>
Answer: A man holds a beach photo of three smiling children, symbolizing parental grief and remembrance following the Newtown tragedy.

---

Now answer for the following image and text:

TEXT: {query}  
IMAGE: <image>""",

"RefCOCO-Matching": """Given an image and a referring expression, your task is to first reason about what object or region the expression refers to, then generate a detailed description of that object or region, and finally write a short summarization of the object or region. Follow the rules below:

- Generate a brief reasoning process (1–2 sentences) that identifies what and where the expression refers to.
- Wrap your reasoning in <think>...</think>.
- Then output a detailed description of the referred object or region using: Answer: <answer>
- Only describe the referred object/region and the background directly behind it. Do not describe any other parts of the image.

---

EXAMPLE

QUERY EXPRESSION: bowl behind the others can only see part  

<think>The expression refers to the partially visible bowl in the back-right of the image, mostly hidden behind other dishes. It appears to be a white ceramic bowl with vertical ridges and a light blue band near its base.</think>  
Answer: A white, rib-sided ceramic bowl with a soft blue band near its base, only its upper rim and curved side visible; it rests on a light-colored cloth atop the table, with muted countertop surface behind.

---

Now answer for the following referring expression:  
QUERY EXPRESSION: {query}
QUERY IMAGE: <image>""",

}


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch


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
        tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
 
    if "internvl" in args.model_name.lower():
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    sampling_params = SamplingParams(max_tokens=1024, stop_token_ids=stop_token_ids)

    for idx, subset in enumerate(args.subset_name):

        print(f"\033[91m{idx+1}/{len(args.subset_name)}: Processing {subset} now!\033[0m")
        # if args.prompt_format == 'cot':
        #     reasoning_prefix, description_prefix = prefix_keys[subset][model_args.model_name]
        dataset = load_dataset(args.dataset_name, subset, split=args.split_name[0])
        qry_image_field = "qry_image_path" if "qry_image_path" in dataset.column_names else "qry_img_path"
        qry_text_field = "qry" if "qry" in dataset.column_names else "qry_text"
        tgt_image_field = "tgt_img_path" if "tgt_img_path" in dataset.column_names else "pos_image_path"
        tgt_text_field = "tgt_text" if "tgt_text" in dataset.column_names else "pos_text"

        if args.encode_target == "query":
            key_image_field = qry_image_field
            key_text_field = qry_text_field
            process_fn = process_fn_qry
            prompt = prompt_cot_inference[subset]
        else:
            key_image_field = tgt_image_field
            key_text_field = tgt_text_field
            process_fn = process_fn_tgt
            prompt = prompt_cot_inference_target[subset]

        if args.dataset_split != "test":
            dataset = dataset.to_pandas()
            dataset = dataset.drop_duplicates(subset=[key_text_field, key_image_field])
            dataset = datasets.Dataset.from_pandas(dataset)

        if args.dataset_split == "test" and args.encode_target == "target":
            paired_dataset = set()
            for row in dataset:
                for text, image in zip(row[key_text_field], row[key_image_field]):
                    paired_dataset.add((text, image))
            
            paired_dataset = sorted(list(paired_dataset))
            dataset = datasets.Dataset.from_dict({
                key_text_field: [x[0] for x in paired_dataset],
                key_image_field: [x[1] for x in paired_dataset]
            })

        if args.n_partitions > 1:
            dataset = dataset.shard(num_shards=args.n_partitions, index=args.current_partition-1)

        image_folder = args.image_dir

        folder = os.path.join("descriptions",  subset, "cot") if args.encode_target == "query" else os.path.join("descriptions_target", subset, "cot")
        os.makedirs(folder, exist_ok=True)

        # load existing descriptions
        if args.encode_target == "target":
            pkl_files = [x for x in os.listdir(folder) if x.startswith("target") and x.endswith(".pkl")]
        else:
            pkl_files =  [x for x in os.listdir(folder) if x.endswith(".pkl")]
        descriptions = {}
        if len(pkl_files) > 0:
            print(f"Found existing descriptions in {folder}, loading...")
            for f in pkl_files:
                descriptions.update(pickle.load(open(os.path.join(folder, f), "rb")))
    
        if args.encode_target == "target":
            intermediate_files = [x for x in os.listdir(folder) if x.startswith("target") and x.endswith(".jsonl")]
        else:
            intermediate_files = [x for x in os.listdir(folder) if x.endswith(".jsonl")]
        if len(intermediate_files) > 0:
            for f in intermediate_files:
                for line in open(os.path.join(folder, f), "r"):
                    line = json.loads(line)
                    descriptions[(line["qry_text"], line["qry_image"])] = line["response"]

        dataset_unprocessed_idx = []

        for idx, row in enumerate(dataset):

            key_text = row[key_text_field]
            key_image = row[key_image_field]
                
            if (key_text, key_image) not in descriptions:
                dataset_unprocessed_idx.append(idx)
        
        dataset = dataset.select(dataset_unprocessed_idx)
                
        print(dataset)
        print(f"Processing {len(dataset)} images in {subset} for partition {args.current_partition}/{args.n_partitions}")

        intermediates = open(os.path.join(folder, f"{args.encode_target}_intermediates_{args.current_partition}-{args.n_partitions}_{str(os.environ.get('SLURM_JOB_ID'))}.jsonl"), "a") 
        
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), args.batch_size)):

                batch = dataset[i:i + args.batch_size]

                qry_texts, qry_images = batch[key_text_field], batch[key_image_field]
                qry_images = [x[0] for x in qry_images] if isinstance(qry_images[0], list) else qry_images
                qry_texts = [x[0] for x in qry_texts] if isinstance(qry_texts[0], list) else qry_texts

                queries = [process_fn(qry, subset) for qry in qry_texts]
                    
                loaded_qry_images = [Image.open(os.path.join(image_folder, qry_image)) if qry_image else None for qry_image in qry_images ]
                inputs = [(prompt.format(query=q), qry_image) for q, qry_image in zip(queries, loaded_qry_images)]

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
                
        pickle.dump(descriptions, open(os.path.join(folder, f"{args.encode_target}_descriptions_{args.dataset_split}_{args.current_partition}-{args.n_partitions}.pkl"), "wb"))
        intermediates.close()
        print(f"Finished processing {subset} for partition {args.current_partition}/{args.n_partitions}.")

if __name__ == "__main__":
    main()
