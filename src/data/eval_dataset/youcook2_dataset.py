import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text
from ..loader.mixed_dataset import AutoPairEvalDataset
from ..prompts import VIDEO_EMBED_INSTRUCTION, TEXT_EMBED_INSTRUCTION


TASK_INST_QRY = "Find a video that demonstrates the following action while making a recipe:"
TASK_INST_TGT = "Understand the content of the provided video."
# TASK_INST_QRY = ""
# TASK_INST_TGT = ""

DATASET_PARSER_NAME = "youcook2"
# slightly less than the official one: https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/csv/validation_youcook.csv?plain=1
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction("YouCook2",
    {'query': TEXT_EMBED_INSTRUCTION,
     'target': VIDEO_EMBED_INSTRUCTION})
class YouCook2EvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__("youcook2", *args,
                         query_key_text="sentence", query_key_mm=None, cand_key_text=None, cand_key_mm="id", **dataset_config)

    def _load_hf_dataset(self):
        return load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']]), None

    def _process_one_sample(self, data_idx, batch_dict, **kwargs):
        image_resolution = kwargs["image_resolution"]
        num_frames       = kwargs["num_frames"]
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]
        model_backbone   = kwargs["model_backbone"]

        # Fields
        video_name      = batch_dict["id"][data_idx]
        rel_video_path  = batch_dict["video_path"][data_idx]
        text            = batch_dict["sentence"][data_idx]

        # Paths
        abs_video_path = os.path.join(video_root, os.path.basename(rel_video_path))
        frame_dir      = os.path.join(frame_root, video_name)

        # Ensure frames and sample
        save_frames(video_path=abs_video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Query: text only
        query_text  = process_input_text(TASK_INST_QRY, model_backbone, text=text)
        query_image = None

        # Candidate: video frames + target text
        cand_text = [process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)]
        cand_image = [{
            "bytes": [None] * len(frame_paths),
            "paths": frame_paths,
            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(frame_paths),
        }]

        dataset_info = {
            "query_id": (text,),
            "cand_names": [video_name],
            "label_name": video_name,
        }

        target_description = None
        if self.target_descriptions:
            target_description = self.target_descriptions.get((video_name,))
            if target_description is None:
                print(f"Missing target description for video {video_name} for {self.dataset_config['dataset_name']} dataset")

        return {
            "query_text": query_text,
            "query_image": query_image,
            "cand_text": cand_text,
            "cand_image": cand_image,
            "dataset_infos": dataset_info,
            "query_description": None,
            "target_description": target_description,
        }

