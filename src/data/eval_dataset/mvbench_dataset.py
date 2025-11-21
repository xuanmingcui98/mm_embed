import cv2
import os
import shutil
import imageio
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

import datasets
from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import MMEBV2EvalDatasetProcessor
from src.data.utils.vision_utils import process_video_frames, qa_template
from src.model.processor import VLM_VIDEO_TOKENS
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_QA_INSTRUCTION, format_qa_with_choices
from ..loader.mixed_dataset import AutoPairEvalDataset

def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token} {prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query

subset_meta = {
    "episodic_reasoning": {"video_path": "tvqa/frames_fps3_hq/", "data_type": "frame", "has_start_end": True},
    "action_sequence": {"video_path": "star/Charades_v1_480/", "data_type": "video", "has_start_end": True},
    "action_prediction": {"video_path": "star/Charades_v1_480/", "data_type": "video", "has_start_end": True},
    "action_antonym": {"video_path": "ssv2_video/", "data_type": "video", "has_start_end": False},
    "fine_grained_action": {"video_path": "Moments_in_Time_Raw/videos/", "data_type": "video", "has_start_end": False},
    "unexpected_action": {"video_path": "FunQA_test/test/", "data_type": "video", "has_start_end": False},
    "object_existence": {"video_path": "clevrer/video_validation/", "data_type": "video", "has_start_end": False},
    "object_interaction": {"video_path": "star/Charades_v1_480/", "data_type": "video", "has_start_end": True},
    "object_shuffle": {"video_path": "perception/videos/", "data_type": "video", "has_start_end": False},
    "moving_direction": {"video_path": "clevrer/video_validation/", "data_type": "video", "has_start_end": False},
    "action_localization": {"video_path": "sta/sta_video/", "data_type": "video", "has_start_end": True},
    "scene_transition": {"video_path": "scene_qa/video/", "data_type": "video", "has_start_end": False},
    "action_count": {"video_path": "perception/videos/", "data_type": "video", "has_start_end": False},
    "moving_count": {"video_path": "clevrer/video_validation/", "data_type": "video", "has_start_end": False},
    "moving_attribute": {"video_path": "clevrer/video_validation/", "data_type": "video", "has_start_end": False},
    "state_change": {"video_path": "perception/videos/", "data_type": "video", "has_start_end": False},
    "fine_grained_pose": {"video_path": "nturgbd/", "data_type": "video", "has_start_end": False},
    "character_order": {"video_path": "perception/videos/", "data_type": "video", "has_start_end": False},
    "egocentric_navigation": {"video_path": "vlnqa/", "data_type": "video", "has_start_end": False},
    "counterfactual_inference": {"video_path": "clevrer/video_validation/", "data_type": "video", "has_start_end": False}
}


def get_index(bound, fps, max_frame, num_segments, first_idx=0):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def read_video(transform, video_path, bound=None):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    images_group = list()
    frame_indices = get_index(bound, fps, max_frame, first_idx=0)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    return torch_imgs

def read_gif(transform, video_path, bound=None, fps=25):
    gif = imageio.get_reader(video_path)
    max_frame = len(gif) - 1

    images_group = list()
    frame_indices = get_index(bound, fps, max_frame, first_idx=0)
    for index, frame in enumerate(gif):
        if index in frame_indices:
            img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            img = Image.fromarray(img)
            images_group.append(img)
    torch_imgs = transform(images_group)
    return torch_imgs

def read_frame(transform, video_path, bound=None, fps=3):
    max_frame = len(os.listdir(video_path))
    images_group = list()
    frame_indices = get_index(bound, fps, max_frame, first_idx=1)  # frame_idx starts from 1
    for frame_index in frame_indices:
        img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
        images_group.append(img)
    torch_imgs = transform(images_group)
    return torch_imgs


decord_method_map = {'video': read_video, 'gif': read_gif, 'frame': read_frame}
TASK_INST_QRY = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: "
TASK_INST_TGT = "Represent the following text:\n"
# TASK_INST_QRY = ""
# TASK_INST_TGT = ""

DATASET_PARSER_NAME = "mvbench"
DATASET_HF_PATH = "OpenGVLab/MVBench"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction("MVBench",
    {'query': VIDEO_QA_INSTRUCTION,
     'target': TEXT_EMBED_INSTRUCTION})
class MVBenchEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

    def _load_hf_dataset(self):
        subsets = []
        for subset_name in subset_meta.keys():
            dataset = load_dataset(DATASET_HF_PATH, subset_name, split="train")
            new_column = [subset_name] * len(dataset)
            dataset = dataset.add_column("subset", new_column)
            subsets.append(dataset)
        dataset = datasets.concatenate_datasets(subsets)

        return dataset, None

    def _process_one_sample(self, data_idx, batch_dict, *args, **kwargs):
        model_backbone   = kwargs["model_backbone"]
        image_resolution = kwargs["image_resolution"]  # unused here (kept None in resolutions to mirror original)
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]
        num_frames       = kwargs["num_frames"]

        subset         = batch_dict["subset"][data_idx]
        query_raw      = batch_dict["question"][data_idx]
        video_filename = batch_dict["video"][data_idx]
        cands_raw      = batch_dict["candidates"][data_idx]
        answer_raw     = batch_dict["answer"][data_idx]


        q_with_token = process_query(
            query_raw,
            prompt=TASK_INST_QRY,
            video_token=VLM_VIDEO_TOKENS[model_backbone],
        )
        query_text, cand_text, answer, answer_idx = qa_template(q_with_token, cands_raw, answer_raw)
        # _, cand_text, answer, answer_idx = qa_template(query_raw, cands_raw, answer_raw)
        # query_text = format_qa_with_choices(query_raw, cand_text)

        # Resolve paths & materialize frames if needed
        subset_info = subset_meta[subset]
        data_type   = subset_info["data_type"]       # "video" or "frame"
        src_rel     = subset_info["video_path"]

        video_path = f"{video_root}/{src_rel}/{video_filename}"
        frame_dir  = f"{frame_root}/{subset}/{video_filename}"

        if data_type == "video" and (not os.path.exists(frame_dir) or not os.listdir(frame_dir)):
            os.makedirs(frame_dir, exist_ok=True)
            assert os.path.exists(video_path), f"video is not found: {video_path}"
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames_saved)
            frame_idx = 0
            saved_frames = 0
            while saved_frames < max_frames_saved and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read() 
                if not ret:
                    break
                frame_path = os.path.join(frame_dir, f"{saved_frames:04d}.jpeg")
                cv2.imwrite(frame_path, frame)
                saved_frames += 1
                frame_idx += step
            cap.release()

        if data_type == "frame" and (not os.path.exists(frame_dir) or not os.listdir(frame_dir)):
            # Source already a directory of frames; mirror to frame_dir
            shutil.copytree(video_path, frame_dir, dirs_exist_ok=True)

        # Sample/query frames for the QUERY side (video context lives on the query)
        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        query_image = {
            "bytes": [None] * len(qry_frame_paths),
            "paths": qry_frame_paths,
            "resolutions": [None] * len(qry_frame_paths),
        }

        # Candidates are text-only (no images)
        cand_image = [None] * len(cand_text)

        query_description = None
        if self.query_descriptions:
            query_description = self.query_descriptions.get((query_raw, subset, video_filename))
            if not query_description:
                print(f'No query description found for ({query_raw}, {video_filename}) for dataset {self.dataset_config["dataset_name"]}')

        dataset_info = {
            "query_id": (query_raw, subset, video_filename),
            "subset": subset,
            "video_id": video_filename,
            "query": query_text,
            "cand_names": cand_text,
            "answer": answer,
            "label_name": answer,
            "answer_idx": answer_idx,
            "qry_frame_paths": qry_frame_paths,
            "num_unique_qry_frame": len(set(qry_frame_paths)),
        }

        return {
            "query_text": query_text,      # raw string; parent will apply chat template if enabled
            "query_image":  query_image,   # dict with paths/bytes/resolutions
            "cand_text":    cand_text,     # list[str]
            "cand_image":   cand_image,    # list[None], aligned with cand_text
            "dataset_infos": dataset_info, # per-sample metadata
            "query_description": query_description,
            "target_description": None,
        }