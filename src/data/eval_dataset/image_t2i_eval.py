from src.data.eval_dataset.base_eval_dataset import MMEBEvalDatasetProcessor
from ..loader.mixed_dataset import AutoPairDataset
from ..prompts import TEXT_EMBED_INSTRUCTION, IMAGE_EMBED_INSTRUCTION

DATASET_PARSER_NAME = "image_t2i"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction(["MSCOCO_t2i", "VisualNews_t2i"],
    {'query': TEXT_EMBED_INSTRUCTION ,
    'target': IMAGE_EMBED_INSTRUCTION})
class ImageT2IEvalDatasetProcessor(MMEBEvalDatasetProcessor):
    def __init__(self, *args, **dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)