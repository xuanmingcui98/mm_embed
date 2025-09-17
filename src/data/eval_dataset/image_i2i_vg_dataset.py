import os
import sys

from src.data.eval_dataset.base_eval_dataset import MMEBEvalDatasetProcessor
from ..prompts import IMAGE_EMBED_INSTRUCTION, TEXT_EMBED_INSTRUCTION
from ..loader.mixed_dataset import AutoPairEvalDataset

DATASET_PARSER_NAME = "image_i2i_vg"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction(["MSCOCO", "Visual7W-Pointing", "RefCOCO"],
    {'query': """Given an image and a query, identify the object or region in the image that the query refers to, and generate a concise description of the object or region.\n\nQuery: {text}\n\nEmbed the object with the answer.""",
    'target': IMAGE_EMBED_INSTRUCTION})
@AutoPairEvalDataset.register_instruction(["RefCOCO-Matching"],
    {'query': """Given an image and a query, identify the object or region in the image that the query refers to, and generate a concise description of the object or region.\n\nQuery: {text}\n\nEmbed the object with the answer.""",
    'target': TEXT_EMBED_INSTRUCTION })
class ImageI2IVGEvalDatasetProcessor(MMEBEvalDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)
