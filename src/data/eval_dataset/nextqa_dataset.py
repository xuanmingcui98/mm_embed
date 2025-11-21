import os
import cv2

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import MMEBV2EvalDatasetProcessor
from src.data.utils.vision_utils import process_video_frames, load_frames, qa_template
from src.model.processor import VLM_VIDEO_TOKENS
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_QA_INSTRUCTION, format_qa_with_choices
from ..loader.mixed_dataset import AutoPairEvalDataset

def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token} {prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query


TASK_INST_QRY = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: "
TASK_INST_TGT = "Represent the following text:\n"
# TASK_INST_QRY = ""
# TASK_INST_TGT = ""

OPTIONS = ['A', 'B', 'C', 'D']


DATASET_PARSER_NAME = "nextqa"
DATASET_HF_PATH = "lmms-lab/NExTQA"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction("NExTQA",
    {'query': VIDEO_QA_INSTRUCTION,
     'target': TEXT_EMBED_INSTRUCTION})
class NextQAEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)
        
    def _load_hf_dataset(self):
        dataset = load_dataset(DATASET_HF_PATH, "MC", split="test")
        return dataset, None


    def _process_one_sample(self, data_idx, batch_dict, *args, **kwargs):
        model_backbone   = kwargs["model_backbone"]
        image_resolution = kwargs["image_resolution"]   # kept unused; resolutions left as None to mirror original
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]
        num_frames       = kwargs["num_frames"]

        # Fields
        video_id = batch_dict["video"][data_idx]
        query    = batch_dict["question"][data_idx]
        answer   = batch_dict["answer"][data_idx]      # index
        qid      = batch_dict["qid"][data_idx]
        _type    = batch_dict["type"][data_idx]
        a0       = batch_dict["a0"][data_idx]
        a1       = batch_dict["a1"][data_idx]
        a2       = batch_dict["a2"][data_idx]
        a3       = batch_dict["a3"][data_idx]
        a4       = batch_dict["a4"][data_idx]

        options = [a0, a1, a2, a3, a4]

        q_with_token = process_query(
            query,
            prompt=TASK_INST_QRY,
            video_token=VLM_VIDEO_TOKENS[model_backbone],
        )
        query_text, cand_text, _, answer_idx = qa_template(q_with_token, options, answer)

        # _, cand_text, _, answer_idx = qa_template(query, options, answer)
        # query_text = format_qa_with_choices(query, cand_text)

        # Paths and frame extraction (if missing)
        video_path = f"{video_root}/{video_id}.mp4"
        frame_dir  = f"{frame_root}/{video_id}"

        frames = load_frames(frame_dir)
        if not frames:
            os.makedirs(frame_dir, exist_ok=True)
            assert os.path.exists(video_path), f"Video not found: {video_path}"
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

        # Candidates are text-only
        cand_image = [None] * len(cand_text)

        dataset_info = {
            "query_id": (query, video_id),
            "question_id": qid,
            "video_id": video_id,
            "query": query_text,
            "cand_names": options,                # keep original options as names
            "answer": options[answer],            # original index maps into options
            "label_name": options[answer],
            "answer_idx": answer_idx,             # from qa_template
            "type": _type,
            "qry_frame_paths": qry_frame_paths,
        }

        query_description = None
        if self.query_descriptions:
            query_description = self.query_descriptions.get((query, video_id))
            if not query_description:
                print(f'No query description found for ({query}, {video_id}) for dataset {self.dataset_config["dataset_name"]}')

        return {
            "query_text": query_text,      # string
            "query_image": query_image,    # dict with paths/bytes/resolutions
            "cand_text": cand_text,        # list[str]
            "cand_image": cand_image,      # list[None]
            "dataset_infos": dataset_info, # dict
            "query_description": query_description,
            "target_description": None,
        }
