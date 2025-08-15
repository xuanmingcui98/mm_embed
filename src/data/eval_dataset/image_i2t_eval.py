from src.data.eval_dataset.base_eval_dataset import MMEBEvalDatasetProcessor
from ..loader.mixed_dataset import AutoPairDataset
from ..prompts import TEXT_EMBED_INSTRUCTION, IMAGE_EMBED_INSTRUCTION

DATASET_PARSER_NAME = "image_i2t"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction(["VisualNews_i2t", "MSCOCO_i2t"],
    {'query': IMAGE_EMBED_INSTRUCTION,
    'target': TEXT_EMBED_INSTRUCTION })
class ImageI2TEvalDatasetProcessor(MMEBEvalDatasetProcessor):
    def __init__(self, *args, **dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)
