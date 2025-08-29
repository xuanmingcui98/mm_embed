import os
import cv2
from datasets import load_dataset

from src.data.eval_dataset.base_eval_dataset import MMEBV2EvalDatasetProcessor
from src.data.utils.vision_utils import process_video_frames, load_frames
from src.model.processor import VLM_VIDEO_TOKENS
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_QA_INSTRUCTION, format_qa_with_choices
from ..loader.mixed_dataset import AutoPairEvalDataset

def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token} {prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query


# TASK_INST_QRY = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: "
# TASK_INST_TGT = "Represent the following text:\n"
TASK_INST_QRY = ""
TASK_INST_TGT = ""
    
OPTIONS = ['A', 'B', 'C', 'D']

DATASET_PARSER_NAME = "videomme"
DATASET_HF_PATH = "lmms-lab/Video-MME"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction("Video-MME",
    {'query': VIDEO_QA_INSTRUCTION,
     'target': TEXT_EMBED_INSTRUCTION})
class VideoMMMEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args,
                         query_key_text='question', query_key_mm='videoID', cand_key_text=None, cand_key_mm='options',
                         **dataset_config)

    def _load_hf_dataset(self):
        return load_dataset(DATASET_HF_PATH, split="test"), None

    def _process_one_sample(self, data_idx, batch_dict, *args, **kwargs):
        model_backbone   = kwargs["model_backbone"]
        image_resolution = kwargs["image_resolution"]   # not used; keep resolutions=None to match original
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]
        num_frames       = kwargs["num_frames"]

        # Fields
        query        = batch_dict["question"][data_idx]
        video_id     = batch_dict["videoID"][data_idx]
        options      = batch_dict["options"][data_idx]
        answer       = batch_dict["answer"][data_idx]        # e.g., 'A','B',...
        question_id  = batch_dict["question_id"][data_idx]
        domain       = batch_dict["domain"][data_idx]
        sub_category = batch_dict["sub_category"][data_idx]

        query_text = format_qa_with_choices(query, options)

        # Paths & frame extraction (if missing)
        video_path = f"{video_root}/{video_id}.mp4"
        frame_dir  = f"{frame_root}/{video_id}"
        frames = load_frames(frame_dir)
        if not frames:
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

        # Sample frames for the QUERY side
        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        query_image = {
            "bytes": [None] * len(qry_frame_paths),
            "paths": qry_frame_paths,
            "resolutions": [None] * len(qry_frame_paths),
        }

        # Candidates: text-only multiple choice; strip "X. " prefix if present
        cand_text  = [o[o.find(". "):].strip(". ") for o in options]
        cand_image = [None] * len(options)

        # Map answer token (e.g., 'A') to index and label
        ans_idx   = OPTIONS.index(answer)
        label_str = options[ans_idx]

        query_description = None
        if self.query_descriptions:
            query_description = self.query_descriptions.get((query, video_id), None)
            if query_description is None:
                print(f"No query description for video {video_id} in {self.dataset_config['dataset_name']} dataset")

        dataset_info = {
            "question_id": question_id,
            "video_id": video_id,
            "query": query_text,
            "cand_names": options,
            "answer": answer,
            "label_name": label_str,
            "answer_idx": ans_idx,
            "domain": domain,
            "sub_category": sub_category,
            "qry_frame_paths": qry_frame_paths,
        }

        return {
            "query_text": query_text,      # str
            "query_image": query_image,    # dict with paths/bytes/resolutions
            "cand_text": cand_text,        # list[str]
            "cand_image": cand_image,      # list[None]
            "dataset_infos": dataset_info, # dict
            "query_description": query_description,
            "target_description": None,
        }