import os

from datasets import load_dataset
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.utils.dataset_utils import load_hf_dataset

from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, MMEBV2EvalDatasetProcessor
from src.data.utils.vision_utils import load_frames, save_frames
from src.model.processor import process_input_text
from ..prompts import (VIDEO_EMBED_INSTRUCTION, 
                       IMAGE_TEXT_EMBED_INSTRUCTION, 
                       VIDEO_TEXT_EMBED_INSTRUCTION, 
                       TEXT_EMBED_INSTRUCTION)
from ..loader.mixed_dataset import AutoPairEvalDataset

TASK_INST_QRY_TEXT = "" # "Find the clip that corresponds to the given text:"
TASK_INST_QRY_IMG = "" # "Select the video clip that aligns with the given text and image:"
TASK_INST_QRY_VIDEO = "" # "Find the clip that corresponds to the given sentence and video segment:"
TASK_INST_TGT = "" # "Understand the content of the provided video clip."

DATASET_PARSER_NAME = "momentseeker"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction("MomentSeeker",
    {'query': {"video": VIDEO_TEXT_EMBED_INSTRUCTION, "image": IMAGE_TEXT_EMBED_INSTRUCTION, "text": TEXT_EMBED_INSTRUCTION},
     'target': VIDEO_EMBED_INSTRUCTION})
class MomentSeekerEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args,
                         query_key_text='query', query_key_mm='input_frames',
                         cand_key_text=None, cand_key_mm='positive_frames',
                         **dataset_config)

    def _load_hf_dataset(self):
        if self.dataset_config.get("data_path", None) != None:
            dataset = load_dataset("json", data_files=self.dataset_config["data_path"])
            dataset = dataset["train"]
        else:
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']])

        return dataset, None

    def _process_one_sample(self, data_idx, batch_dict, *args, **kwargs):
        image_resolution   = kwargs["image_resolution"]
        num_negative_clips = kwargs["num_negative_clips"] 
        num_video_frames   = kwargs["num_video_frames"]
        model_backbone     = kwargs["model_backbone"]
        video_root         = kwargs.get("video_root")
        clip_root          = kwargs.get("clip_root")
        frame_root         = kwargs.get("frame_root")

        # Pull this sample's fields
        query          = batch_dict["query"][data_idx]
        positive_frames = batch_dict["positive_frames"][data_idx]   # list of dicts with "output_path"
        negative_frames = batch_dict["negative_frames"][data_idx]   # list of dicts with "output_path"
        input_frames    = batch_dict["input_frames"][data_idx]      # e.g., "*.mp4", "*.jpg", or text handle

        # Defaults
        query_text  = None
        query_image = None
        cand_text, cand_image = [], []
        cand_clip_names = []
        pos_clip_name   = []  # list of positives (kept as list to mirror original)

        # --- Build query text/image depending on input modality ---
        if isinstance(input_frames, str) and input_frames.endswith(".mp4"):
            # Text: video query (no tokens; parent chat-template will handle tokens if enabled) 

            if self.data_args.apply_chat_template:
                query_text = self.instruction['query']['video'].format(text=query)
            else:
                query_text = process_input_text(TASK_INST_QRY_VIDEO, model_backbone, text=query, add_video_token=True)

            # Frames dir for this query video
            query_video_name = input_frames.split(".mp4")[0].replace("/", "_")
            query_frame_dir  = os.path.join(frame_root, "video_frames", query_video_name)

            query_description = self.query_descriptions[(query, os.path.join("video_frames", query_video_name))] if self.query_descriptions else None

            # Extract frames if needed, then load
            if not os.path.exists(query_frame_dir):
                query_video_path = os.path.join(video_root, input_frames)
                save_frames(video_path=query_video_path,
                            frame_dir=query_frame_dir,
                            max_frames_saved=num_video_frames)
            qry_frame_paths = load_frames(query_frame_dir)

            query_image = {
                "bytes": [None] * len(qry_frame_paths),
                "paths": qry_frame_paths,
                "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_frame_paths),
            }

        elif isinstance(input_frames, str) and input_frames.endswith(".jpg"):
            # Text: image query

            if self.data_args.apply_chat_template:
                query_text = self.instruction['query']['image'].format(text=query)
            else:
                query_text = process_input_text(TASK_INST_QRY_IMG, model_backbone, text=query, add_image_token=True)

            # Use the provided single image (stored under frame_root as "query_<fname>")
            input_image_path = os.path.join(frame_root, f"query_{input_frames}")
            query_image = {
                "bytes": [None],
                "paths": [input_image_path],
                "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)],
            }

        else:
            # Pure text query

            if self.data_args.apply_chat_template:
                query_text = self.instruction['query']['text'].format(text=query)
            else:
                query_text = process_input_text(TASK_INST_QRY_TEXT, model_backbone, text=query, add_video_token=True)
            query_image = None

        # --- Build candidate clips (positives + negatives) ---
        pos_clip_paths = [entry["output_path"] for entry in positive_frames]
        neg_clip_paths = [entry["output_path"] for entry in negative_frames]

        target_description = []

        # Helper to add one clip candidate
        def _add_clip_candidate(path: str, is_positive: bool):
            clip_name = path.replace("/", "_").split(".mp4")[0]
            clip_frame_dir = os.path.join(frame_root, "video_frames", clip_name)

            if not os.path.exists(clip_frame_dir):
                clip_abs = os.path.join(clip_root, path)
                save_frames(video_path=clip_abs, frame_dir=clip_frame_dir, max_frames_saved=num_video_frames)

            frame_paths = load_frames(clip_frame_dir)
            cand_image.append({
                "bytes": [None] * len(frame_paths),
                "paths": frame_paths,
                "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(frame_paths),
            })

            if self.target_descriptions:
                target_description.append(self.target_descriptions[(os.path.join("video_frames", clip_name),)])
            cand_clip_names.append(clip_frame_dir)
            if is_positive:
                pos_clip_name.append(clip_frame_dir)

        for p in pos_clip_paths:
            _add_clip_candidate(p, is_positive=True)
        for n in neg_clip_paths:
            _add_clip_candidate(n, is_positive=False)

        # One target text replicated for all candidates (parent will wrap with chat template if enabled)
        if (len(pos_clip_paths) + len(neg_clip_paths)) > 0:
            tgt_txt = process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)
            cand_text = [tgt_txt] * (len(pos_clip_paths) + len(neg_clip_paths))

        dataset_infos = {
            "cand_names": cand_clip_names,
            "label_name": pos_clip_name,   # list of positive clip frame dirs
        }

        return {
            "query_text": query_text,
            "query_image": query_image,
            "cand_text": cand_text,
            "cand_image": cand_image,
            "dataset_infos": dataset_infos,
            "query_description": query_description,
            "target_description": target_description,
        }