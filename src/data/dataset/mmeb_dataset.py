from ..loader.mixed_dataset import AutoPairDataset
from ..prompts import IMAGE_TASKS, TASK_TYPE
from .base_pair_dataset import BaseDatasetProcessor
from ..prompts import (TEXT_EMBED_INSTRUCTION, 
                       IMAGE_EMBED_INSTRUCTION, 
                       IMAGE_QA_INSTRUCTION,
                       VISDOC_QA_RETRIEVAL_INSTRUCTION)

  

DATASET_PARSER_NAME = "mmeb"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction([task for task in IMAGE_TASKS if task in TASK_TYPE['vqa']],
    {'query': IMAGE_QA_INSTRUCTION,
    'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction(["ImageNet-1K", "ImageNet_1K", "ImageNet-A", "ImageNet-R", "VOC2007", "ObjectNet"], 
    {'query': """Given the image, identify the main object in the image.\n\nEmbed your answer.""",
    'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction(["Country211"],
    {'query': """Given an image, identify the country where it was taken.\n\nEmbed your answer.""",
   'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction(["HatefulMemes"],
    {'query': """Given an image, determine if it contains hateful speech.\n\nEmbed your answer.""",
   'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction(["SUN397", "Place365"],
    {'query': """Given an image, identify the scene it depicts.\n\nEmbed your answer.""",
   'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction("N24News",
    {'query': """Given an image and the below news text, identify the main domain of the news.\n\nNews text: {text}\n\nEmbed your answer.""",
    'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction("VisDial",
    {'query': """Given the dialogue about an image, generate a description of the image based on the dialogue.\n\n{text}\n\nEmbed your answer.""",
   'target': IMAGE_EMBED_INSTRUCTION})
@AutoPairDataset.register_instruction("CIRR",
    {'query': """Given a base image and a modification instruction of how to modify the base image to get the target image, generate a description of the target image.\n\nModification instruction: {text}\n\nEmbed the image with the instruction and your answer.""",
   'target': IMAGE_EMBED_INSTRUCTION})
@AutoPairDataset.register_instruction("FashionIQ",
    {'query': """Given a garment image and a modification instruction of how to modify the garment in the base garment image to get the target garment, generate a description of the target garment.\n\nEmbed the image with the instruction and your answer.""",
   'target': """Given an garment image, generate a detailed description, and embed these content."""})
@AutoPairDataset.register_instruction(["MSCOCO_t2i", "VisualNews_t2i"],
    {'query': TEXT_EMBED_INSTRUCTION ,
    'target': IMAGE_EMBED_INSTRUCTION})
@AutoPairDataset.register_instruction(["VisualNews_i2t", "MSCOCO_i2t"],
    {'query': IMAGE_EMBED_INSTRUCTION,
    'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction(["MSCOCO", "Visual7W-Pointing", "RefCOCO"],
    {'query': """Given an image and a query, identify the object or region in the image that the query refers to, and generate a concise description of the object or region.\n\nQuery: {text}\n\nEmbed the object with the answer.""",
    'target': IMAGE_EMBED_INSTRUCTION})
@AutoPairDataset.register_instruction(["RefCOCO-Matching"],
    {'query': """Given an image and a query, identify the object or region in the image that the query refers to, and generate a concise description of the object or region.\n\nQuery: {text}\n\nEmbed the object with the answer.""",
    'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction(["WebQA", "OVEN"],
    {'query': """Given a question, determine what kind of image-text pair would help answer the question.\n\nQuestion: {text}\n\nEmbed your answer.""",
    'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction("EDIS",
    {'query': """Given a news text, determine a similar image-text pair.\n\nEmbed your answer.""",
    'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction("Wiki-SS-NQ",
    {'query': VISDOC_QA_RETRIEVAL_INSTRUCTION,
    'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction("NIGHTS",
    {'query': IMAGE_EMBED_INSTRUCTION,
    'target': IMAGE_EMBED_INSTRUCTION})
class MMEBDatasetProcessor(BaseDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config,
                         query_key_text="qry",
                         query_key_mm="qry_image_path",
                         cand_key_text="pos_text",
                         cand_key_mm="pos_image_path")

        self.dataset_name = self.dataset_config.get("dataset_name")