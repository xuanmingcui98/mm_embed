from ..loader.mixed_dataset import AutoPairDataset, AutoSFTDataset
from ..prompts import IMAGE_TASKS, TASK_TYPE
from .base_pair_dataset import BaseDatasetProcessor, BaseSFTDatasetProcessor
from ..prompts import (TEXT_EMBED_INSTRUCTION, 
                       IMAGE_EMBED_INSTRUCTION, 
                       IMAGE_QA_INSTRUCTION,
                       VISDOC_QA_RETRIEVAL_INSTRUCTION)

  

DATASET_PARSER_NAME = "mmeb"
@AutoPairDataset.register(DATASET_PARSER_NAME)

class MMEBDatasetProcessor(BaseDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

        self.dataset_name = self.dataset_config.get("dataset_name")

@AutoSFTDataset.register(DATASET_PARSER_NAME)
class MMEBSFTDatasetProcessor(BaseSFTDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)