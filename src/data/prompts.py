import re
from typing import List

IMAGE_TASKS = {'ImageNet-1K', "ImageNet_1K", 'N24News', 'HatefulMemes', 'VOC2007', 'SUN397', 'A-OKVQA', 'MSCOCO', 'Place365', 'ImageNet-A', 'ImageNet-R', 'ObjectNet', 'Country211', 'OK-VQA', 'RefCOCO', 'DocVQA', 'InfographicsVQA', 'ChartQA', 'NIGHTS', 'FashionIQ', 'ScienceQA', 'Visual7W', 'VizWiz', 'GQA', 'TextVQA', 'VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 'MSCOCO_i2t', 'Wiki-SS-NQ', 'WebQA', 'OVEN', 'EDIS', 'RefCOCO-Matching', 'Visual7W-Pointing'}
VIDEO_TASKS = {"video_caption_300k-v2t", "video_qa_240k", "video_caption_300k-t2v", # train
               'SmthSmthV2', 'HMDB51', 'UCF101', 'Kinetics-700', 'Breakfast', 'MSR-VTT', 'MSVD', 'DiDeMo', 'YouCook2', 'VATEX', 'QVHighlight', 'Charades-STA', 'MomentSeeker', 'Video-MME', 'NExTQA', 'EgoSchema', 'MVBench', 'ActivityNetQA'}
VISDOC_TASKS = {"VisRAG-Indomain-data", "colpali_train_set",
                'ViDoRe_arxivqa', 'ViDoRe_docvqa', 'ViDoRe_infovqa', 'ViDoRe_tabfquad', 'ViDoRe_tatdqa', 'ViDoRe_shiftproject', 'ViDoRe_syntheticDocQA_artificial_intelligence', 'ViDoRe_syntheticDocQA_energy', 'ViDoRe_syntheticDocQA_government_reports', 'ViDoRe_syntheticDocQA_healthcare_industry', 'VisRAG_ArxivQA', 'VisRAG_ChartQA', 'VisRAG_MP-DocVQA', 'VisRAG_SlideVQA', 'VisRAG_InfoVQA', 'VisRAG_PlotQA', 'ViDoSeek-page', 'ViDoSeek-doc', 'MMLongBench-doc', 'MMLongBench-page', "ViDoRe_esg_reports_human_labeled_v2", "ViDoRe_biomedical_lectures_v2_multilingual", "ViDoRe_economics_reports_v2_multilingual", "ViDoRe_esg_reports_v2_multilingual"}

ALL_TASKS = IMAGE_TASKS | VIDEO_TASKS | VISDOC_TASKS

ALL_EVAL_IMAGE_TASKS = IMAGE_TASKS - {"ImageNet_1K"}
ALL_EVAL_VIDEO_TASKS = VIDEO_TASKS - {"video_caption_300k", "video_qa_240k", "video_caption_300k-video"}  # remove train tasks
ALL_EVAL_VISDOC_TASKS = VISDOC_TASKS - {"VisRAG-Indomain-data", "colpali_train_set"}  # remove train tasks

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


# TEXT_EMBED_INSTRUCTION  = """Embed the following text:\n\n{text}"""
TEXT_EMBED_INSTRUCTION = """Represent the following text as text embeddings:\n\n{text}"""
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



query_user_prompts_cot = {}

for task in IMAGE_TASKS:
    if task in TASK_TYPE['vqa']:
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


query_user_prompts_cot_for_generation = {**query_user_prompts_cot}
for task in ["MSCOCO", "Visual7W-Pointing", "RefCOCO", "RefCOCO-Matching"]:
    query_user_prompts_cot_for_generation[task] = """Given an image and a query object, generate a detailed description of the query object and the region where it is located in the image, then generate a summary. Focus solely on the object and the region it is located. Do not mention anything else in the image. Let's think step by step.

Query: {query}"""

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
    elif task in TASK_TYPE['vqa'] | TASK_TYPE['classification']:
        target_user_prompts_cot[task] = """Represent the following text as text embeddings.
Answer: {query}"""


HMDB51_LABELS = ['ride_bike', 'wave', 'dive', 'pour', 'smile', 'eat', 'shoot_ball', 'clap', 'chew', 'brush_hair', 'pushup', 'draw_sword', 'pullup', 'catch', 'somersault', 'sit', 'hug', 'kick_ball', 'golf', 'cartwheel', 'turn', 'kick', 'dribble', 'jump', 'sword_exercise', 'drink', 'shoot_gun', 'hit', 'ride_horse', 'smoke', 'fencing', 'climb', 'situp', 'handstand', 'talk', 'kiss', 'push', 'shake_hands', 'swing_baseball', 'stand', 'flic_flac', 'sword', 'run', 'laugh', 'walk', 'throw', 'punch', 'climb_stairs', 'pick', 'shoot_bow', 'fall_floor']
UCF101_LABELS = ['Kayaking', 'PlayingDaf', 'BoxingSpeedBag', 'GolfSwing', 'Diving', 'PlayingFlute', 'TrampolineJumping', 'BabyCrawling', 'Drumming', 'RockClimbingIndoor', 'Biking', 'MoppingFloor', 'Haircut', 'SalsaSpin', 'PlayingPiano', 'HandstandWalking', 'Billiards', 'Nunchucks', 'SoccerJuggling', 'VolleyballSpiking', 'Skijet', 'LongJump', 'UnevenBars', 'ApplyEyeMakeup', 'SoccerPenalty', 'BandMarching', 'BlowingCandles', 'PizzaTossing', 'ApplyLipstick', 'HandstandPushups', 'CricketShot', 'FrisbeeCatch', 'PushUps', 'FieldHockeyPenalty', 'ThrowDiscus', 'PlayingGuitar', 'JugglingBalls', 'HorseRiding', 'PlayingViolin', 'JumpingJack', 'PoleVault', 'BoxingPunchingBag', 'FloorGymnastics', 'RopeClimbing', 'JumpRope', 'ParallelBars', 'BodyWeightSquats', 'HorseRace', 'BasketballDunk', 'BlowDryHair', 'SkyDiving', 'BalanceBeam', 'IceDancing', 'SumoWrestling', 'Bowling', 'TennisSwing', 'MilitaryParade', 'Lunges', 'Swing', 'HighJump', 'StillRings', 'Skiing', 'HeadMassage', 'JavelinThrow', 'HammerThrow', 'Hammering', 'BaseballPitch', 'Shotput', 'Basketball', 'PommelHorse', 'Punch', 'TableTennisShot', 'SkateBoarding', 'Typing', 'Rafting', 'WritingOnBoard', 'PlayingSitar', 'Archery', 'PlayingTabla', 'TaiChi', 'BreastStroke', 'ShavingBeard', 'CricketBowling', 'Rowing', 'CliffDiving', 'CleanAndJerk', 'PullUps', 'PlayingDhol', 'YoYo', 'FrontCrawl', 'WallPushups', 'WalkingWithDog', 'Knitting', 'Mixing', 'HulaHoop', 'BrushingTeeth', 'Surfing', 'PlayingCello', 'BenchPress', 'CuttingInKitchen', 'Fencing']
BREAKFAST_LABELS = ['pancake', 'cereal', 'sandwich', 'scrambledegg', 'friedegg', 'coffee', 'milk', 'tea', 'juice', 'salad']

query_user_prompts_cot_generation = {**query_user_prompts_cot}
target_user_prompts_cot_generation = {**target_user_prompts_cot}

for task in ["MSCOCO", "Visual7W-Pointing", "RefCOCO", "RefCOCO-Matching"]:
    query_user_prompts_cot_generation[task] = """Given an image and a query object, generate a detailed description of the object and the region where it is located in the image, then generate a summary of the description. Focus solely on the object and the region it is located. Do not mention anything else in the image. Let's think step by step.

Query: {query}"""

for task in VIDEO_TASKS | VISDOC_TASKS:
    if task in {"Kinetics-700"}:
        query_user_prompts_cot_generation[task] = """Given a video, identify the main action or activity being performed. Let's think step by step."""
    elif task in {"SmthSmthV2"}:
        query_user_prompts_cot_generation[task] = """Given a video, identify the actions or object interactions being performed by the person in the video. Let's think step by step."""
    elif task in {"UCF101"}:
        query_user_prompts_cot_generation[task] = """Given a video, identify the activities or sports being performed by the person in the video. Let's think step by step."""
    elif task in {"HMDB51"}:
        query_user_prompts_cot_generation[task] = """Given a video, identify the actions or objects interactions being performed by the person in the video. Let's think step by step."""
    elif task in {"Breakfast"}:
        query_user_prompts_cot_generation[task] = """Given a video, recognize the breakfast type that the person is cooking in the video. Let's think step by step."""
    elif task in {"colpali_train_set"}:
        query_user_prompts_cot_generation[task] = """Given the image and the below question, answer the question based on the image. Let's think step by step.

Question: {query}"""
    elif task in {"QVHighlight", "Charades-STA"}:
        query_user_prompts_cot_generation[task] = """Given an image and a query describing a specific clip in the video, generate a detailed description of the specific clip the query refers to, then generate a summary of the description. Focus solely on the clip. Do not mention anything else in the video. Let's think step by step.

Query: {query}"""
    elif task in {"MomentSeeker"}:
        query_user_prompts_cot_generation[task] = {
            "text": """Given a text query, generate a detailed description of the target video clip that the query could refer to, then generate a summary of the description. Let's think step by step.
            
Query: {query}""",

            "image": """Given an image and a text query, generate a detailed description of the target video clip that the image could refer to, then generate a summary of the description. Let's think step by step.
            
Query: {query}""",
            "video": """Given a video and a text query, generate a detailed description of the target video clip that the query could refer to, then generate a summary of the description. Let's think step by step.
            
Query: {query}"""
        }

    elif task in {'Video-MME', 'NExTQA', 'EgoSchema', 'MVBench', 'ActivityNetQA'}:
        query_user_prompts_cot_generation[task] = """Given a video and a question about the video, answer the question based on the video content. Choose the answer from the provided options. Let's think step by step.

Question: {query}"""

    elif task in {"video_caption_300k-v2t"}:
        query_user_prompts_cot_generation[task] = """Given a video, generate a detailed description of the video content, then generate a summarization based on the description. Let's think step by step."""

    elif task in {"video_qa_240k"}:
        query_user_prompts_cot_generation[task] = """Given a video and a question about the video, answer the question based on the video content. Let's think step by step.

Question: {query}"""

### Target side
    
    elif task in {"video_caption_300k-t2v", "MSR-VTT", "MSVD", "YouCook2", "VATEX", "DiDeMo"}:
        target_user_prompts_cot_generation[task] = """Given a video, generate a detailed description of the video content, then generate a summarization based on the description. Let's think step by step."""
    elif task in VISDOC_TASKS:
        target_user_prompts_cot_generation[task] = """Given a document image, first generate a detailed description of the document content, then generate a summarization based on the description. Let's think step by step."""
    