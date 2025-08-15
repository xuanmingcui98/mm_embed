import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_EMBED_INSTRUCTION
from ..loader.mixed_dataset import AutoPairDataset

# TASK_INST_QRY = "Select a video that fits the description provided:"
# TASK_INST_TGT = "Understand the content of the provided video."
TASK_INST_QRY = ""
TASK_INST_TGT = ""

DATASET_PARSER_NAME = "vatex"
# 4,478 example since a lot of videos are not valid
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction("VATEX",
    {'query': TEXT_EMBED_INSTRUCTION,
    'target': VIDEO_EMBED_INSTRUCTION})
class VatexEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args,
                         query_key_text="query_key_text", query_key_mm="videoID",
                         cand_key_text=None, cand_key_mm="videoID",
                         **dataset_config)
        
    def _load_hf_dataset(self):
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']])
        return dataset, None

    def _process_one_sample(self, data_idx, batch_dict, *args, **kwargs):
        image_resolution = kwargs["image_resolution"]
        num_frames       = kwargs["num_frames"]
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]
        model_backbone   = kwargs["model_backbone"]

        # Fields
        video_name = batch_dict["videoID"][data_idx]
        captions   = batch_dict["enCap"][data_idx]
        caption_text = captions[0] if isinstance(captions, (list, tuple)) and len(captions) > 0 else captions

        # Paths
        video_path = os.path.join(video_root, f"{video_name}.mp4")
        frame_dir  = os.path.join(frame_root, video_name)

        # Ensure frames exist and sample
        save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Query: text only
        query_text  = process_input_text(TASK_INST_QRY, model_backbone, text=caption_text)
        query_image = None

        # Candidate: video frames + target text
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
            "query_text": query_text,      # str
            "query_image": query_image,    # None
            "cand_text": cand_text,        # list[str]
            "cand_image": cand_image,      # list[dict]
            "dataset_infos": dataset_info, # dict
        }