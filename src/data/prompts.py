import re
from typing import List

IMAGE_TASKS = {'ImageNet-1K', "ImageNet_1K", 'N24News', 'HatefulMemes', 'VOC2007', 'SUN397', 'A-OKVQA', 'MSCOCO', 'Place365', 'ImageNet-A', 'ImageNet-R', 'ObjectNet', 'Country211', 'OK-VQA', 'RefCOCO', 'DocVQA', 'InfographicsVQA', 'ChartQA', 'NIGHTS', 'FashionIQ', 'ScienceQA', 'Visual7W', 'VizWiz', 'GQA', 'TextVQA', 'VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 'MSCOCO_i2t', 'Wiki-SS-NQ', 'WebQA', 'OVEN', 'EDIS', 'RefCOCO-Matching', 'Visual7W-Pointing'}
VIDEO_TASKS = {"video_caption_300k-v2t", "video_qa_240k", "video_caption_300k-t2v", # train
               'SmthSmthV2', 'HMDB51', 'UCF101', 'Kinetics-700', 'Breakfast', 'MSR-VTT', 'MSVD', 'DiDeMo', 'YouCook2', 'VATEX', 'QVHighlight', 'Charades-STA', 'MomentSeeker', 'Video-MME', 'NExTQA', 'EgoSchema', 'MVBench', 'ActivityNetQA'}
VISDOC_TASKS = {"VisRag-Indomain-data", "colpali_train_set",
                'ViDoRe_arxivqa', 'ViDoRe_docvqa', 'ViDoRe_infovqa', 'ViDoRe_tabfquad', 'ViDoRe_tatdqa', 'ViDoRe_shiftproject', 'ViDoRe_syntheticDocQA_artificial_intelligence', 'ViDoRe_syntheticDocQA_energy', 'ViDoRe_syntheticDocQA_government_reports', 'ViDoRe_syntheticDocQA_healthcare_industry', 'VisRAG_ArxivQA', 'VisRAG_ChartQA', 'VisRAG_MP-DocVQA', 'VisRAG_SlideVQA', 'VisRAG_InfoVQA', 'VisRAG_PlotQA', 'ViDoSeek-page', 'ViDoSeek-doc', 'MMLongBench-doc', 'MMLongBench-page', "ViDoRe_esg_reports_human_labeled_v2", "ViDoRe_biomedical_lectures_v2_multilingual", "ViDoRe_economics_reports_v2_multilingual", "ViDoRe_esg_reports_v2_multilingual"}

ALL_TASKS = IMAGE_TASKS | VIDEO_TASKS | VISDOC_TASKS

ALL_EVAL_IMAGE_TASKS = IMAGE_TASKS - {"ImageNet_1K"}
ALL_EVAL_VIDEO_TASKS = VIDEO_TASKS - {"video_caption_300k", "video_qa_240k", "video_caption_300k-video"}  # remove train tasks
ALL_EVAL_VISDOC_TASKS = VISDOC_TASKS - {"VisRag-Indomain-data", "colpali_train_set"}  # remove train tasks

TRAIN_TASKS = ['ImageNet_1K', 'N24News', 'HatefulMemes', 'VOC2007', 'SUN397', 'OK-VQA', 'A-OKVQA', 'DocVQA', 'InfographicsVQA', 'ChartQA', 'Visual7W', 'VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 'MSCOCO_i2t', 'NIGHTS', 'WebQA', 'MSCOCO', 'vidore/colpali_train_set', 'VisRag-Indomain-data', 'video_caption_300k-v2t', 'video_caption_300k-t2v', 'video_qa_240k']

VIDORE_QA_RETRIEVAL_DATASETS = [
    "ViDoRe_arxivqa",
    "ViDoRe_docvqa",
    "ViDoRe_infovqa",
    "ViDoRe_tabfquad",
    "ViDoRe_tatdqa",
    "ViDoRe_shiftproject",
    "ViDoRe_syntheticDocQA_artificial_intelligence",
    "ViDoRe_syntheticDocQA_energy",
    "ViDoRe_syntheticDocQA_government_reports",
    "ViDoRe_syntheticDocQA_healthcare_industry",
    "ViDoRe_esg_reports_human_labeled_v2",
    "ViDoRe_biomedical_lectures_v2_multilingual",
    "ViDoRe_economics_reports_v2_multilingual",
    "ViDoRe_esg_reports_v2_multilingual",

    "ViDoSeek-page",
    "ViDoSeek-doc",
    "MMLongBench-doc",
    "MMLongBench-page"
]

VISRAG_QA_RETRIEVAL_DATASETS = [
    "VisRAG_ArxivQA",
    "VisRAG_ChartQA",
    "VisRAG_MP-DocVQA",
    "VisRAG_SlideVQA",
    "VisRAG_InfoVQA",
    "VisRAG_PlotQA"
]


TASK2ID = dict(zip(ALL_TASKS, range(len(ALL_TASKS))))

TASK_TYPE = {
    "classification": {"ImageNet-1K", "ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"},
    "vqa": {"OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"},
    "retrieval": {"VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"},
    "grounding": {"MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"}
}


TEXT_EMBED_INSTRUCTION  = """Embed the following text:\n\n{text}"""
IMAGE_EMBED_INSTRUCTION = """Given the image, generate a detailed description.\n\nEmbed the image with the description."""
VISDOC_EMBED_INSTRUCTION = """Given the document image, generate a detailed description of the document.\n\nEmbed the document image with the description."""
VIDEO_EMBED_INSTRUCTION = """Given the video, generate a detailed description of the video.\n\nEmbed the video with the description."""
VIDEO_QA_INSTRUCTION = """Given the video and the below question, answer the question based on the video.\n\nQuestion: {text}\n\nEmbed your answer."""
IMAGE_QA_INSTRUCTION = """Given the image and the below question, answer the question based on the image.\n\nQuestion: {text}\n\nEmbed your answer."""
VISDOC_QA_RETRIEVAL_INSTRUCTION = """Given a question, determine the visual document that would help answer the question.\n\nQuestion: {text}\n\nEmbed your answer."""

IMAGE_TEXT_EMBED_INSTRUCTION = """Given the image and text, generate a detailed description of the image. Text: {text}\n\nEmbed the image and text with the description."""
VIDEO_TEXT_EMBED_INSTRUCTION = """Given the video and text, generate a detailed description of the video. Text: {text}\n\nEmbed the video and text with the description."""

def format_description(description, prompt_format="cot_answer"):
    if not description:
        return ""
    
    # sometime the teacher generation is not clean.
    if "<think>" in description:
        index = description.find("<think>")
        description = description[index:]
    if prompt_format == "answer_only":
        return "Answer: " + description.split("Answer:")[-1].strip(". \n")
    elif prompt_format == 'cot_only':
        return description.split("Answer:")[0].strip(". \n")
    else:
        return description.strip(". \n")

def extract_query(qry, subset):
    if qry is None:
        return ""
    
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
    elif subset in TASK_TYPE["vqa"]:
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
    else:
        return qry


def extract_target(text, subset):
    if text is None:
        return ""
    if subset in {"WebQA", "OVEN"}:
        text = text.replace("Represent the given Wikipedia image with related text information: ", "")
    elif subset in {"EDIS"}:
        text = text.replace("Represent the given image with related text information: ", "")
    elif subset in {"RefCOCO-Matching"}:
        text = text.replace("Select the portion of the image that follows the language expressions: ", "")
    return text.replace("<|image_1|>", "").strip()


def format_qa_with_choices(query, choices: List[str]):
    return query + "\nOptions:\n" + "\n".join(choices)