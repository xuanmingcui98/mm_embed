from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import os, pickle
from datasets.features.image import image_to_bytes

from torch.jit import isinstance
from ..dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.utils import print_master, print_rank
from torch.utils.data import Dataset
from ..prompts import IMAGE_TASKS, TASK_TYPE
from functools import partial
import torch
from .base_pair_dataset import BaseDatasetProcessor
from ..prompts import (TEXT_EMBED_INSTRUCTION, 
                       IMAGE_EMBED_INSTRUCTION, 
                       VISDOC_EMBED_INSTRUCTION, 
                       VIDEO_EMBED_INSTRUCTION, 
                       VISDOC_RETRIEVAL_INSTRUCTION)



DATASET_PARSER_NAME = "mmeb"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction([task for task in IMAGE_TASKS if task in TASK_TYPE['vqa']],
                                      {'query': """Given the image and the below question, answer the question based on the image.\n\nQuestion: {query}\n\nEmbed your answer.""",
                                        'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction(["ImageNet-1K", "ImageNet_1K", "ImageNet-A", "ImageNet-R", "VOC2007", "ObjectNet"], {
                                        'query': """Given the image, identify the main object in the image.\n\nEmbed your answer.""",
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
                                      {'query': """Given an image and the below news text, identify the main domain of the news.\n\nNews text: {query}\n\nEmbed your answer.""",
                                       'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction("VisDial",
                                      {'query': """Given the dialogue about an image, generate a description of the image based on the dialogue.\n\n{query}\n\nEmbed your answer.""",
                                       'target': IMAGE_EMBED_INSTRUCTION})
@AutoPairDataset.register_instruction("CIRR",
                                      {'query': """Given a base image and a modification instruction of how to modify the base image to get the target image, generate a description of the target image.\n\nModification instruction: {query}\n\nEmbed the image with the instruction and your answer.""",
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
                                      {'query': """Given an image and a query, identify the object or region in the image that the query refers to, and generate a concise description of the object or region.\n\nQuery: {query}\n\nEmbed the object with the answer.""",
                                       'target': IMAGE_EMBED_INSTRUCTION})
@AutoPairDataset.register_instruction(["RefCOCO-Matching"],
                                      {'query': """Given an image and a query, identify the object or region in the image that the query refers to, and generate a concise description of the object or region.\n\nQuery: {query}\n\nEmbed the object with the answer.""",
                                       'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction(["WebQA", "OVEN"],
                                      {'query': """Given a question, determine what kind of image-text pair would help answer the question.\n\nQuestion: {query}\n\nEmbed your answer.""",
                                       'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction("EDIS",
                                      {'query': """Given a news text, determine a similar image-text pair.\n\nEmbed your answer.""",
                                       'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction("Wiki-SS-NQ",
                                      {'query': VISDOC_RETRIEVAL_INSTRUCTION,
                                        'target': TEXT_EMBED_INSTRUCTION })
@AutoPairDataset.register_instruction("NIGHTS",
                                      {'query': IMAGE_EMBED_INSTRUCTION,
                                       'target': IMAGE_EMBED_INSTRUCTION})
class MMEBDatasetProcessor(BaseDatasetProcessor):
    def __init__(self, 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 instruction,
                 **dataset_config):
        
        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, processor, instruction,
                         query_key_text="qry",
                         query_key_mm="qry_image_path",
                         cand_key_text="pos_text",
                         cand_key_mm="pos_image_path",
                         **dataset_config)