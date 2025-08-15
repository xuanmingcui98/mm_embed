import os
import sys

from datasets import load_dataset

from src.data.eval_dataset.base_eval_dataset import MMEBV2EvalDatasetProcessor
from src.data.utils.vision_utils import process_video_frames, load_frames
from src.model.processor import VLM_VIDEO_TOKENS
import cv2
from ..prompts import (TEXT_EMBED_INSTRUCTION, VIDEO_QA_INSTRUCTION)
from ..loader.mixed_dataset import AutoPairDataset

def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token} {prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query


TASK_PROMPT = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: "
OPTIONS = ['A', 'B', 'C', 'D']

DATASET_PARSER_NAME = "egoschema"
DATASET_HF_PATH = "lmms-lab/egoschema"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction("EgoSchema",
    {'query': VIDEO_QA_INSTRUCTION,
     'target': TEXT_EMBED_INSTRUCTION})
class EgoSchemaEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args, **dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args,
                         query_key_text="question", query_key_mm="video_idx",
                         cand_key_text="option", cand_key_mm=None,
                         **dataset_config)
        
    def _load_hf_dataset(self):
        return load_dataset(DATASET_HF_PATH, "Subset", split="test"), None

    def _process_one_sample(self, data_idx, batch_dict, *args, **kwargs):
        model_backbone   = kwargs["model_backbone"]
        image_resolution = kwargs["image_resolution"]  # unused here (res kept as None to match original)
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]
        num_frames       = kwargs["num_frames"]

        # Pull this sample's fields
        video_idx    = batch_dict["video_idx"][data_idx]
        query        = batch_dict["question"][data_idx]
        answer_idx   = int(batch_dict["answer"][data_idx])
        question_idx = batch_dict["question_idx"][data_idx]
        options      = batch_dict["option"][data_idx]

        # Build query text (original concatenates options into the prompt)
        query = process_query(
            query + " " + " ".join(options),
            prompt=TASK_PROMPT,
            video_token=VLM_VIDEO_TOKENS[model_backbone],
        )

        # Paths
        video_path = f"{video_root}/{video_idx}.mp4"
        frame_dir  = f"{frame_root}/{video_idx}"

        # Ensure frames exist (extract via OpenCV if missing)
        frames = load_frames(frame_dir)
        if not frames:
            print(f"Extracting frames for: {video_path}")
            os.makedirs(frame_dir, exist_ok=True)
            assert os.path.exists(video_path)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames_saved)
            frame_idx = 0
            saved_frames = 0
            while saved_frames < max_frames_saved:
                assert cap.isOpened(), "not cap.isOpened()"
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(frame_dir, f"{saved_frames:04d}.jpeg")
                cv2.imwrite(frame_path, frame)
                saved_frames += 1
                frame_idx += step
            cap.release()
            print(f"[{DATASET_PARSER_NAME}] Extracted #frames: {saved_frames}, dumped to {frame_dir}")

        # Sample/query frames for the query side (video context lives on the query)
        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        query_image = {
            "bytes": [None] * len(qry_frame_paths),
            "paths": qry_frame_paths,
            "resolutions": [None] * len(qry_frame_paths),
        }

        # Candidates: text-only multiple choice; no images
        cand_text  = [o[o.find(". "):].strip(". ") for o in options]  # mirrors original slicing
        cand_image = [None] * len(options)

        dataset_info = {
            "question_id": question_idx,
            "video_id": video_idx,
            "query": query,
            "cand_names": options,
            "answer": options[answer_idx],
            "label_name": options[answer_idx],
            "answer_idx": answer_idx,
            "qry_frame_paths": qry_frame_paths,
        }

        return {
            "query_text": query,        # raw (parent will apply chat template if enabled)
            "query_image": query_image, # dict with paths/bytes/resolutions; parent detects video via len(paths)>1
            "cand_text": cand_text,     # list[str]
            "cand_image": cand_image,   # list[None], zipped with cand_text
            "dataset_infos": dataset_info,
        }