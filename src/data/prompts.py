import re

IMAGE_TASKS = ['ImageNet-1K', "ImageNet_1K", 'N24News', 'HatefulMemes', 'VOC2007', 'SUN397', 'A-OKVQA', 'MSCOCO', 'Place365', 'ImageNet-A', 'ImageNet-R', 'ObjectNet', 'Country211', 'OK-VQA', 'RefCOCO', 'DocVQA', 'InfographicsVQA', 'ChartQA', 'NIGHTS', 'FashionIQ', 'ScienceQA', 'Visual7W', 'VizWiz', 'GQA', 'TextVQA', 'VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 'MSCOCO_i2t', 'Wiki-SS-NQ', 'WebQA', 'OVEN', 'EDIS', 'RefCOCO-Matching', 'Visual7W-Pointing']

# VIDEO_TASKS = ['SmthSmthV2', 'HMDB51', 'UCF101', 'Kinetics-700', 'Breakfast', 'MSR-VTT', 'MSVD', 'DiDeMo', 'YouCook2', 'VATEX', 'QVHighlight', 'Charades-STA', 'MomentSeeker', 'Video-MME', 'NExTQA', 'EgoSchema', 'MVBench', 'ActivityNetQA']

VIDEO_TASKS = ["SmthSmthV2", "HMDB51", "Kinetics-700", "UCF101", "Breakfast", 
               "vidore/colpali_train_set", "openbmb/VisRAG-Ret-Train-In-domain-data", 
               "video_caption_300k", "video_qa_240k", "MSR-VTT", "video_caption_300k-video"]
VISDOC_TASKS = ['ViDoRe_arxivqa', 'ViDoRe_docvqa', 'ViDoRe_infovqa', 'ViDoRe_tabfquad', 'ViDoRe_tatdqa', 'ViDoRe_shiftproject', 'ViDoRe_syntheticDocQA_artificial_intelligence', 'ViDoRe_syntheticDocQA_energy', 'ViDoRe_syntheticDocQA_government_reports', 'ViDoRe_syntheticDocQA_healthcare_industry', 'VisRAG_ArxivQA', 'VisRAG_ChartQA', 'VisRAG_MP-DocVQA', 'VisRAG_SlideVQA', 'VisRAG_InfoVQA', 'VisRAG_PlotQA', 'ViDoSeek-page', 'ViDoSeek-doc', 'MMLongBench-doc', 'MMLongBench-page']

ALL_TASKS = IMAGE_TASKS + VIDEO_TASKS + VISDOC_TASKS

TASK2ID = dict(zip(ALL_TASKS, range(len(ALL_TASKS))))

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

def extract_query_from_mmeb(qry, subset):
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


def extract_target_from_mmeb(text, subset):
    if subset in {"WebQA", "OVEN"}:
        text = text.replace("Represent the given Wikipedia image with related text information: ", "")
    elif subset in {"EDIS"}:
        text = text.replace("Represent the given image with related text information: ", "")
    elif subset in {"RefCOCO-Matching"}:
        text = text.replace("Select the portion of the image that follows the language expressions: ", "")
    
    return text.replace("<|image_1|>", "").strip()


def format_text_for_chat_template(processor, text, image_path, video_path=None, description=None, add_generation_prompt=False):

    formatted_sample = [
        {"role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],}
    ]
    user_content = [] 
    if image_path:
        user_content.append({"type": "image", "image": image_path})
    if video_path:
        user_content.append({"type": "video", "video": video_path})
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

for task in IMAGE_TASKS:
    if task in task_categories['vqa']:
        query_user_prompts_cot[task] = """Given the image and the below question, answer the question based on the image. Let's think step by step.

Question: {query}"""
    elif task in {"ImageNet-1K", "ImageNet_1K", "ImageNet-A", "ImageNet-R", "VOC2007", "ObjectNet"}:
        query_user_prompts_cot[task] = """Given an image, identify the main object category it belongs to. Let's think step by step."""
    elif task in {"Country211"}:
        query_user_prompts_cot[task] = """Given an image, identify the country where it was taken. Let's think step by step."""
    elif task in {"HatefulMemes"}:
        query_user_prompts_cot[task] = """Given an image, determine if it contains hateful speech. Identify the texts in the image. Let's think step by step."""
    elif task in {"SUN397", "Place365"}:
        query_user_prompts_cot[task] = """Given an image, identify the scene it depicts. Let's think step by step."""
    elif task in {"N24News"}:
        query_user_prompts_cot[task] = """Given an image and its associated news text, identify the main domain of the news. Let's think step by step.

News text: {query}"""
    elif task in {"VisDial"}:
        query_user_prompts_cot[task] = """Given the dialogue about an image, generate a description of the image based on the dialogue. Let's think step by step.

Dialogue: {query}"""
    elif task in {"CIRR"}:
        query_user_prompts_cot[task] = """Given a base image and a modification instruction of how to modify the base image to get a target image, generate a description of the target image. Let's think step by step.

Modification instruction: {query}"""

    elif task in {"FashionIQ"}:
        query_user_prompts_cot[task] = """Given a base fashion image and a modification instruction of how to modify the garment in the base fashion image to get the target garment, generate a description of the target fashion image. Focus solely on the garment and ignore the background or the person in the image. Let's think step by step.

Modification instruction: {query}"""

    elif task in {"VisualNews_t2i"}:
        query_user_prompts_cot[task] = """Given the news text, first use your knowledge to expand it into a more detailed and concise description of the target news image, then generate a summarization based on the description. Let's think step by step.

News text: {query}"""

    elif task in {"VisualNews_i2t"}:
        query_user_prompts_cot[task] = """Given a news image, first generate a concise and informative description of the news image, then generate a summarization based on the description. Let's think step by step."""

    elif task in {"MSCOCO_t2i"}:
        query_user_prompts_cot[task] = """Given a COCO-style caption, first use your knowledge to expand it into a more detailed and concise description of the target image, then generate a summarization based on the description. Let's think step by step.

Caption: {query}"""
    elif task in {"MSCOCO_i2t"}:
        query_user_prompts_cot[task] = """Given an image, first generate a detailed and informative description of the image, then generate a COCO-style caption based on the description. Let's think step by step."""
    elif task in {"MSCOCO", "Visual7W-Pointing", "RefCOCO", "RefCOCO-Matching"}:
        query_user_prompts_cot[task] = """Given an image and a query, identify the object or region in the image that the query refers to, and generate a concise description of the object or region. Let's think step by step.

Query: {query}"""
    elif task in {"WebQA", "OVEN"}:
        query_user_prompts_cot[task] = """Given a question, determine what kind of image-text pair would help answer the question. Let's think step by step.

Question: {query}"""
    
    elif task in {"EDIS"}:
        query_user_prompts_cot[task] = """Given a news text, expand the caption with more visual and textual cues that would help retrieve a matching image-text pair, then generate a summarization based on the description. Let's think step by step.

News text: {query}"""
    elif task in {"Wiki-SS-NQ"}:
        query_user_prompts_cot[task] = """Given a question, explain what information is needed to answer the question. Let's think step by step.

Question: {query}"""
    elif task in {"NIGHTS"}:
        query_user_prompts_cot[task] = """Given a query image, generate a concise and informative description of what the target image may look like, then generate a summarization based on the description. Let's think step by step."""

query_user_prompts_base = {}

for task in VIDEO_TASKS:
    if task in {"Kinetics-700"}:
        query_user_prompts_base[task] = """Given a video, identify the main action or activity being performed. Let's think step by step."""
    elif task in {"SmthSmthV2"}:
        query_user_prompts_base[task] = """Given a video, identify the actions or object interactions being performed by the person in the video. Let's think step by step."""
    elif task in {"UCF101"}:
        query_user_prompts_base[task] = """Given a video, identify the activities or sports being performed by the person in the video. Let's think step by step."""
    elif task in {"HMDB51"}:
        query_user_prompts_base[task] = """Given a video, identify the actions or objects interactions being performed by the person in the video. Let's think step by step."""
    elif task in {"Breakfast"}:
        query_user_prompts_base[task] = """Given a video, recognize the breakfast type that the person is cooking in the video. Let's think step by step."""
    elif task in {"MSR-VTT"}:
        query_user_prompts_base[task] = """Understand the content of the provided video."""
    

target_user_prompts_cot = {}

for task in IMAGE_TASKS:
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
        target_user_prompts_cot[task] = """Given a fashion image, first generate a detailed description of the garment in the image, then generate a summarization based on the description. Focus solely on the garment and ignore the background or the person in the image. Let's think step by step."""
    elif task in {"VisDial"}:
        target_user_prompts_cot[task] = query_user_prompts_cot['MSCOCO_i2t']
    elif task in {"NIGHTS"}:
        target_user_prompts_cot[task] = query_user_prompts_cot['NIGHTS']
    elif task in {"OVEN", "WebQA"}:
        target_user_prompts_cot[task] = """Given an wiki image and a short text description, first generate a detailed description of the image and text, then generate a summarization based on the description. Let's think step by step.

Text: {query}"""
    elif task in {"Wiki-SS-NQ"}:
        target_user_prompts_cot[task] = """Given a document screenshot, first generate a detailed description of the document content, then generate a summarization based on the description. Let's think step by step."""
    elif task in {"EDIS"}:
        target_user_prompts_cot[task] = """Given a news image and a text description, first generate a detailed description of the news image and text, then generate a summarization based on the description. Let's think step by step.

News text: {query}"""
    elif task in {"MSCOCO", "Visual7W-Pointing", "RefCOCO"}:
        target_user_prompts_cot[task] = """Given an image, first generate a detailed and informative description of the image, and then generate a summarization based on the description. Let's think step by step."""
    elif task in {"RefCOCO-Matching"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["MSCOCO"]
    elif task in task_categories['vqa'] | task_categories['classification']:
        target_user_prompts_cot[task] = """Represent the following text as text embeddings.
Answer: {query}"""

target_user_prompts_base = {}

for task in VIDEO_TASKS:
    if task in {"MSR-VTT"}:
        target_user_prompts_base[task] = """Understand the content of the provided video."""


for task in VIDEO_TASKS:
    if task in {"MSR-VTT"}:
        target_user_prompts_cot[task] = """Understand the content of the provided video."""


def get_query(task, query, use_cot=True):

    query = extract_query_from_mmeb(query, task)

    if task in query_user_prompts_cot:
        query = query_user_prompts_cot[task].format(query=query)
    
    if not use_cot:
        query = query.replace(" Let's think step by step.", "").strip()

    return query

def get_target(task, query, use_cot=True):

    query = extract_target_from_mmeb(query, task)
    if task in target_user_prompts_cot:
        query = target_user_prompts_cot[task].format(query=query)
    
    if not use_cot:
        query = query.replace(" Let's think step by step.", "").strip()
    
    return query
