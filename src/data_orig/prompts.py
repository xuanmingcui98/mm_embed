import re

tasks = ['ImageNet-1K', "ImageNet_1K", 'N24News', 'HatefulMemes', 'VOC2007', 'SUN397', 'A-OKVQA', 'MSCOCO', 
         'Place365', 'ImageNet-A', 'ImageNet-R', 'ObjectNet', 'Country211', 'OK-VQA', 'RefCOCO', 
         'DocVQA', 'InfographicsVQA', 'ChartQA', 'NIGHTS', 'FashionIQ', 'ScienceQA', 'Visual7W', 
         'VizWiz', 'GQA', 'TextVQA', 'VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 
         'MSCOCO_i2t', 'Wiki-SS-NQ', 'WebQA', 'OVEN', 'EDIS', 'RefCOCO-Matching', 'Visual7W-Pointing']

task_categories = {
    "classification": {"ImageNet-1K", "ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"},
    "vqa": {"OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"},
    "retrieval": {"VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"},
    "grounding": {"MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"}
}

def format_description(description, prompt_format="gt_only"):
    if description is None:
        return 
    
    # sometime the teacher generation is not clean.
    if "<think>" in description:
        index = description.find("<think>")
        description = description[index:]
    if prompt_format == "gt_only":
        return "Answer: " + description.split("Answer:")[-1].strip(". \n")
    else:
        return description.strip(". \n")

def extract_query(qry, subset):
    if subset in {"CIRR"}:
        return qry.replace("<|image_1|>\nGiven an image, find a similar everyday image with the described changes: ", "").strip()
    elif subset in {"FashionIQ"}:
        return qry.replace("<|image_1|>\nFind an image to match the fashion image and style note: ", "").strip()
    elif subset in {"EDIS"}:
        return qry.replace("<|image_1|>\nFind a news image that matches the provided caption: ", "").strip()
    elif subset in {"RefCOCO"}:
        return qry.replace("<|image_1|>\nSelect the portion of the image that follows the language expressions. ", "").strip()
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


def extract_target(text, subset):
    if subset in {"WebQA", "OVEN"}:
        text = text.replace("Represent the given Wikipedia image with related text information: ", "")
    elif subset in {"EDIS"}:
        text = text.replace("Represent the given image with related text information: ", "")
    elif subset in {"RefCOCO-Matching"}:
        text = text.replace("Select the portion of the image that follows the language expressions: ", "")
    
    return text.replace("<|image_1|>", "").strip()


def format_text(processor, text, image_path, description=None, add_generation_prompt=False):

    formatted_sample = [
        {"role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],}
    ]
    user_content = [] if not image_path else [{"type": "image", "image": image_path}]
    user_content.append({"type": "text", "text": text})
    formatted_sample.append({"role": "user", "content": user_content})

    if not add_generation_prompt:
        formatted_sample.append({
            "role": "assistant",
            "content": [{"type": "text", "text": description if description is not None else ""}],
        })
    
    formatted_sample = processor.apply_chat_template(formatted_sample, add_generation_prompt=add_generation_prompt, tokenize=False)
    if not add_generation_prompt:
        formatted_sample = formatted_sample.strip()

    return formatted_sample 

query_user_prompts_cot = {}

for task in tasks:
    if task in task_categories['vqa']:
        query_user_prompts_cot[task] = """Given the image and the below question, answer the question based on the image. Explain your reasoning briefly.

Question: {query}"""
    elif task in {"ImageNet-1K", "ImageNet_1K", "ImageNet-A", "ImageNet-R", "VOC2007", "ObjectNet"}:
        query_user_prompts_cot[task] = """Given an image, identify the main object category it belongs to. Explain your reasoning briefly."""
    elif task in {"Country211"}:
        query_user_prompts_cot[task] = """Given an image, identify the country where it was taken. Explain your reasoning briefly."""
    elif task in {"HatefulMemes"}:
        query_user_prompts_cot[task] = """Given an image, determine if it contains hateful speech. Identify the texts in the image. Explain your reasoning briefly."""
    elif task in {"SUN397", "Place365"}:
        query_user_prompts_cot[task] = """Given an image, identify the scene it depicts. Explain your reasoning briefly."""
    elif task in {"N24News"}:
        query_user_prompts_cot[task] = """Given an image and its associated news text, identify the main domain of the news. Explain your reasoning briefly.

News text: {query}"""
    elif task in {"VisDial"}:
        query_user_prompts_cot[task] = """Given the dialogue about an image, generate a description of the image based on the dialogue. Explain your reasoning briefly.

Dialogue: {query}"""
    elif task in {"CIRR"}:
        query_user_prompts_cot[task] = """Given a base image and a modification instruction of how to modify the base image to get a target image, generate a description of the target image. Explain your reasoning briefly.

Modification instruction: {query}"""

    elif task in {"FashionIQ"}:
        query_user_prompts_cot[task] = """Given a base fashion image and a modification instruction of how to modify the garment in the base fashion image to get the target garment, generate a description of the target fashion image. Focus solely on the garment and ignore the background or the person in the image. Explain your reasoning briefly.

Modification instruction: {query}"""

    elif task in {"VisualNews_t2i"}:
        query_user_prompts_cot[task] = """Given the news text, first use your knowledge to expand it into a more detailed and concise description of the target news image, then generate a summarization based on the description. Explain your reasoning briefly.

News text: {query}"""

    elif task in {"VisualNews_i2t"}:
        query_user_prompts_cot[task] = """Given a news image, first generate a concise and informative description of the news image, then generate a summarization based on the description. Explain your reasoning briefly."""

    elif task in {"MSCOCO_t2i"}:
        query_user_prompts_cot[task] = """Given a COCO-style caption, first use your knowledge to expand it into a more detailed and concise description of the target image, then generate a summarization based on the description. Explain your reasoning briefly.

Caption: {query}"""
    elif task in {"MSCOCO_i2t"}:
        query_user_prompts_cot[task] = """Given an image, first generate a detailed and informative description of the image, then generate a COCO-style caption based on the description. Explain your reasoning briefly."""
    elif task in {"MSCOCO", "Visual7W-Pointing", "RefCOCO", "RefCOCO-Matching"}:
        query_user_prompts_cot[task] = """Given an image and a query, identify the object or region in the image that the query refers to, and generate a concise description of the object or region. Explain your reasoning briefly.

Query: {query}"""
    elif task in {"WebQA", "OVEN"}:
        query_user_prompts_cot[task] = """Given a question, determine what kind of image-text pair would help answer the question. Explain your reasoning briefly.

Question: {query}"""
    
    elif task in {"EDIS"}:
        query_user_prompts_cot[task] = """Given a news text, expand the caption with more visual and textual cues that would help retrieve a matching image-text pair, then generate a summarization based on the description. Explain your reasoning briefly.

News text: {query}"""
    elif task in {"Wiki-SS-NQ"}:
        query_user_prompts_cot[task] = """Given a question, explain what information is needed to answer the question. Explain your reasoning briefly.

Question: {query}"""
    elif task in {"NIGHTS"}:
        query_user_prompts_cot[task] = """Given a query image, generate a concise and informative description of what the target image may look like, then generate a summarization based on the description. Explain your reasoning briefly."""


target_user_prompts_cot = {}

for task in tasks:
    if task in {"MSCOCO_i2t"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["MSCOCO_t2i"]
    elif task in {"MSCOCO_t2i"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["MSCOCO_i2t"]
    elif task in {"VisualNews_i2t"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["VisualNews_t2i"]
    elif task in {"VisualNews_t2i"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["VisualNews_i2t"]
    elif task in {"CIRR"}:
        target_user_prompts_cot[task] = query_user_prompts_cot['MSCOCO_i2t']
    elif task in {"FashionIQ"}:
        target_user_prompts_cot[task] = """Given a fashion image, first generate a detailed description of the garment in the image, then generate a summarization based on the description. Focus solely on the garment and ignore the background or the person in the image. Explain your reasoning briefly."""
    elif task in {"VisDial"}:
        target_user_prompts_cot[task] = query_user_prompts_cot['MSCOCO_i2t']
    elif task in {"NIGHTS"}:
        target_user_prompts_cot[task] = query_user_prompts_cot['NIGHTS']
    elif task in {"OVEN", "WebQA"}:
        target_user_prompts_cot[task] = """Given an wiki image and a short text description, first generate a detailed description of the image and text, then generate a summarization based on the description. Explain your reasoning briefly.

Text: {query}"""
    elif task in {"Wiki-SS-NQ"}:
        target_user_prompts_cot[task] = """Given a document screenshot, first generate a detailed description of the document content, then generate a summarization based on the description. Explain your reasoning briefly."""
    elif task in {"EDIS"}:
        target_user_prompts_cot[task] = """Given a news image and a text description, first generate a detailed description of the news image and text, then generate a summarization based on the description. Explain your reasoning briefly.

News text: {query}"""
    elif task in {"MSCOCO", "Visual7W-Pointing", "RefCOCO"}:
        target_user_prompts_cot[task] = """Given an image, first generate a detailed and informative description of the image, and then generate a summarization based on the description. Explain your reasoning briefly."""
    elif task in {"RefCOCO-Matching"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["MSCOCO"]
    elif task in task_categories['vqa'] | task_categories['classification']:
        target_user_prompts_cot[task] = query_user_prompts_cot[task]    

def get_query(task, query, use_cot=True):

    query = extract_query(query, task)

    if task in query_user_prompts_cot:
        query = query_user_prompts_cot[task].format(query=query)
    
    if not use_cot:
        query = query.replace(" Explain your reasoning briefly.", "").strip()

    return query

def get_target(task, query, use_cot=True):

    query = extract_target(query, task)
    if task in target_user_prompts_cot:
        query = target_user_prompts_cot[task].format(query=query)
    
    if not use_cot:
        query = query.replace(" Explain your reasoning briefly.", "").strip()
    
    return query
