import json
import os
import shutil
import sys

from datasets import load_dataset

from src.data.eval_dataset.base_eval_dataset import MMEBV2EvalDatasetProcessor
from src.data.utils.vision_utils import process_video_frames, load_frames
from src.model.processor import VLM_VIDEO_TOKENS
import random
import cv2
from ..prompts import VIDEO_QA_INSTRUCTION, TEXT_EMBED_INSTRUCTION
from ..loader.mixed_dataset import AutoPairDataset, add_metainfo_hook
                       

def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token} {prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query


TASK_INST_QRY = "" # "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: "
TASK_INST_TGT = "" # "Represent the following text:\n{query}"
OPTIONS = ['yes', 'no']


DATASET_PARSER_NAME = "activitynetqa"
DATASET_HF_PATH = "lmms-lab/ActivityNetQA"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction("ActivityNetQA",
    {'query': VIDEO_QA_INSTRUCTION,
     'target': TEXT_EMBED_INSTRUCTION})
class ActivityNetQAEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args, **dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args, 
                        query_key_text="question", query_key_mm="video_name",
                        cand_key_text=None, cand_key_mm=None, **dataset_config)

    def _load_hf_dataset(self):
        return load_dataset('json', data_files=self.dataset_config["data_path"])['train'], None

    def _process_one_sample(self, idx, batch_dict, *args, **kwargs):

        model_backbone = kwargs['model_backbone']
        max_frames_saved = kwargs['max_frames_saved']
        video_root = kwargs['video_root']
        frame_root = kwargs['frame_root']
        num_frames = kwargs['num_frames']

        video_name, query, answer, question_id = \
            batch_dict['video_name'][idx], batch_dict['question'][idx], \
            batch_dict['answer'][idx], batch_dict['question_id'][idx]

        query = process_query(query + '? (A) yes; (B) no.', prompt=TASK_INST_QRY, video_token=VLM_VIDEO_TOKENS[model_backbone])
        
        video_path = f'{video_root}/v_{video_name}.mp4'
        frame_dir = f'{frame_root}/v_{video_name}'
        frames = load_frames(frame_dir)
        if not frames:
            # print(f'Extracting frames for: {video_path}')
            os.makedirs(frame_dir, exist_ok=True)
            assert os.path.exists(video_path)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames_saved)
            frame_idx = 0
            saved_frames = 0
            while saved_frames < max_frames_saved:
                assert cap.isOpened(), "not cap.isOpened()"
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Move to specific frame
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(frame_dir, f"{saved_frames:04d}.jpeg")
                cv2.imwrite(frame_path, frame)
                saved_frames += 1
                frame_idx += step
            cap.release()
            # print(f'[{DATASET_PARSER_NAME}] Extracted #frames: {saved_frames}, dumped to {frame_dir}')

        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        query_images = {"bytes": [None] * len(qry_frame_paths), "paths": qry_frame_paths, "resolutions": [None] * len(qry_frame_paths)}
        cand_images = [None] * len(OPTIONS)
        dataset_info = {
                "question_id": question_id,
                "video_id": video_name,
                "query": query,
                "cand_names": OPTIONS,
                "answer": answer,
                "label_name": answer,
                "answer_idx": OPTIONS.index(answer),
                "qry_frame_paths": qry_frame_paths,
            }

        return {
            "query_text": query,
            "query_image": query_images,
            "cand_text": OPTIONS,
            "cand_image": cand_images,
            "dataset_infos": dataset_info,
            }