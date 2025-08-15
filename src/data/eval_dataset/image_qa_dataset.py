from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, MMEBEvalDatasetProcessor
from ..prompts import TEXT_EMBED_INSTRUCTION, IMAGE_QA_INSTRUCTION, IMAGE_TASKS, TASK_TYPE
from ..loader.mixed_dataset import AutoPairDataset


DATASET_PARSER_NAME = "image_qa"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction([task for task in IMAGE_TASKS if task in TASK_TYPE['vqa']],
    {'query': IMAGE_QA_INSTRUCTION,
    'target': TEXT_EMBED_INSTRUCTION })
class ImageQAEvalDatasetProcessor(MMEBEvalDatasetProcessor):
    def __init__(self, *args, **dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)