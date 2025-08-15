from src.data.eval_dataset.base_eval_dataset import MMEBEvalDatasetProcessor
from ..loader.mixed_dataset import AutoPairDataset
from ..prompts import TEXT_EMBED_INSTRUCTION

DATASET_PARSER_NAME = "image_cls"
@AutoPairDataset.register(DATASET_PARSER_NAME)
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
class ImageClsEvalDatasetProcessor(MMEBEvalDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)