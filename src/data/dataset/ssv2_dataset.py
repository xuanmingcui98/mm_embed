import os
import sys

from datasets import load_dataset
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, MULTIMODAL_FEATURES, VideoDatasetProcessor
from src.data.eval_dataset.video_classification_utils import VIDEOCLS_LABEL_MAPPING, DATASET_INSTRUCTION


DATASET_PARSER_NAME = "ssv2"
@AutoPairDataset.register(DATASET_PARSER_NAME)
class SSV2DatasetProcessor(VideoDatasetProcessor):
    def __init__(self, model_args, data_args, training_args, **kwargs):
        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, **kwargs)
