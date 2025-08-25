import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_EMBED_INSTRUCTION
from ..loader.mixed_dataset import AutoPairEvalDataset

# TASK_INST_QRY = "Find a video that contains the following visual content:"
# TASK_INST_TGT = "Understand the content of the provided video."
TASK_INST_QRY = ""
TASK_INST_TGT = ""

DATASET_PARSER_NAME = "msrvtt"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction("MSR-VTT",
    {'query': TEXT_EMBED_INSTRUCTION,
     'target': VIDEO_EMBED_INSTRUCTION})
class MSRVTTEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args,
                         query_key_text='caption', query_key_mm=None,
                         cand_key_text=None, cand_key_mm='video_id',
                         **dataset_config)

    def _load_hf_dataset(self):
        return load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']]), None

    def _process_one_sample(self, data_idx, batch_dict, **kwargs):
        image_resolution = kwargs["image_resolution"]
        max_frames_saved = kwargs["max_frames_saved"]
        num_frames       = kwargs["num_frames"]
        model_backbone   = kwargs["model_backbone"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]

        # Fields for this sample
        video_name = batch_dict["video_id"][data_idx]
        rel_video_path = batch_dict["video"][data_idx]
        caption = batch_dict["caption"][data_idx]

        # Paths
        abs_video_path = os.path.join(video_root, rel_video_path)
        frame_dir = os.path.join(frame_root, video_name)

        # Ensure frames exist and sample them
        save_frames(video_path=abs_video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Query: text only (no query-side image/video)
        query_text  = process_input_text(TASK_INST_QRY, model_backbone, text=caption)
        query_image = None

        target_description = self.target_descriptions[(video_name,)] if self.target_descriptions else None

        cand_text = [process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)]
        cand_image = [{
            "bytes": [None] * len(frame_paths),
            "paths": frame_paths,
            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(frame_paths),
        }]

        dataset_info = {
            "cand_names": [video_name],
            "label_name": video_name,
        }

        return {
            "query_text": query_text,
            "query_image": query_image,
            "cand_text": cand_text,
            "cand_image": cand_image,
            "dataset_infos": dataset_info,
            "query_description": None,
            "target_description": target_description,
        }
