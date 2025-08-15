import os

from datasets import Dataset
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset
from src.data.eval_dataset.video_classification_utils import DATASET_INSTRUCTION
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text
from ..prompts import TEXT_EMBED_INSTRUCTION
from ..loader.mixed_dataset import AutoPairEvalDataset

# TASK_INST_TGT = "Represent the following text:\n"

TASK_INST_QRY = ""
TASK_INST_TGT = ""

DATASET_PARSER_NAME = "ssv2"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction("SmthSmthV2",
    {'query': """Given the video, identify the actions or object interactions being performed by the person. Embed the video with your answer.""",
    'target': TEXT_EMBED_INSTRUCTION})
class SSV2EvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args,
                         query_key_text=None, query_key_mm='video_id',
                         cand_key_text='pos_text', cand_key_mm=None,
                         **dataset_config)

        self.dataset_config['image_resolution'] = dataset_config.image_resolution

    def _load_hf_dataset(self):
        return load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_name]), None

    def _process_one_sample(self, data_idx, batch_dict, **kwargs):
        image_resolution = kwargs["image_resolution"]
        num_frames       = kwargs["num_frames"]
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]
        dataset_name     = kwargs["dataset_name"]
        model_backbone   = kwargs["model_backbone"]

        # Fields
        video_id  = batch_dict["video_id"][data_idx]
        pos_text  = batch_dict["pos_text"][data_idx]
        cand_list = batch_dict["neg_text"][data_idx]  # list of negatives

        # Paths
        video_path = os.path.join(video_root, f"{video_id}.mp4")
        frame_dir  = os.path.join(frame_root, str(video_id))

        # Ensure frames exist and sample
        save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Query: instruction text (video-token included for non-chat-template path)
        query_text  = process_input_text(DATASET_INSTRUCTION[dataset_name], model_backbone, add_video_token=True)
        query_image = {
            "bytes": [None] * len(frame_paths),
            "paths": frame_paths,
            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(frame_paths),
        }

        # Candidates: text only (negatives), no images
        cand_text  = list(cand_list) if isinstance(cand_list, (list, tuple)) else [cand_list]
        cand_image = [None] * len(cand_text)

        dataset_info = {
            "cand_names": cand_text,
            "label_name": pos_text,
        }

        return {
            "query_text": query_text,
            "query_image": query_image,
            "cand_text": cand_text,
            "cand_image": cand_image,
            "dataset_infos": dataset_info,
        }