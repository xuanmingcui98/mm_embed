import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text
from ..prompts import (TEXT_EMBED_INSTRUCTION, VIDEO_EMBED_INSTRUCTION)
from ..loader.mixed_dataset import AutoPairEvalDataset


TASK_INST_QRY = "Find a video that includes the following described scenes:"
TASK_INST_TGT = "Understand the content of the provided video."
# TASK_INST_QRY = ""
# TASK_INST_TGT = ""

DATASET_PARSER_NAME = "didemo"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction("DiDeMo",
    {'query': TEXT_EMBED_INSTRUCTION,
     'target': VIDEO_EMBED_INSTRUCTION})
class DiDemoEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

    def _load_hf_dataset(self):
        return load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_name]), None

    def _process_one_sample(self, data_idx, batch_dict, **kwargs):
        image_resolution = kwargs["image_resolution"]
        model_backbone   = kwargs["model_backbone"]
        num_frames       = kwargs["num_frames"]
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]

        # Pull this sample's fields
        video_rel = batch_dict["video"][data_idx]
        caption   = batch_dict["caption"][data_idx]

        video_basename = os.path.basename(video_rel)
        video_name     = os.path.splitext(video_basename)[0]
        video_path     = os.path.join(video_root, video_basename)
        frame_dir      = os.path.join(frame_root, video_name)

        # Defaults (parent handles chat-template)
        query_text  = process_input_text(TASK_INST_QRY, model_backbone, text=caption)
        query_image = None
        cand_text, cand_image = [], []
        dataset_infos = {"cand_names": [video_name], "label_name": video_name}

        target_description = None
        if self.target_descriptions:
            target_description = self.target_descriptions.get((video_rel,))
            if not target_description:
                print(f'No target description found for ({video_rel},) for dataset {self.dataset_config["dataset_name"]}')

        try:
            # Extract & process frames
            save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
            frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

            if frame_paths:
                cand_text.append(process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True))
                cand_image.append(
                    ImageVideoInstance(
                        bytes=[None] * len(frame_paths),
                        paths=frame_paths,
                        resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(frame_paths),
                    ).to_dict()
                )
            else:
                dataset_infos["error"] = "No frames returned from process_video_frames."

        except Exception as e:
            dataset_infos["error"] = str(e)

        return {
            "query_text": query_text,
            "query_image": query_image,     # None => no query-side image/video
            "cand_text": cand_text,         # list[str]
            "cand_image": cand_image,       # list[dict], zipped with cand_text
            "dataset_infos": dataset_infos, # per-sample info
            "query_description": None,
            "target_description": target_description,
        }