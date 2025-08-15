import os
import cv2
from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import MMEBV2EvalDatasetProcessor
from src.data.utils.vision_utils import process_video_frames, load_frames, qa_template
from src.model.processor import VLM_VIDEO_TOKENS
import datasets
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_QA_INSTRUCTION
from ..loader.mixed_dataset import AutoPairDataset

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

subset_names = ['Perception', 'Comprehension', 'Adaptation']

DATASET_PARSER_NAME = "videommmu"
DATASET_HF_PATH = "lmms-lab/VideoMMMU"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction("VideoMMMU",
    {'query': VIDEO_QA_INSTRUCTION,
     'target': TEXT_EMBED_INSTRUCTION})
class VideoMMMUEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args,
                         query_key_text='question', query_key_mm="query_key_mm",
                         cand_key_text=None, cand_key_mm=None,
                         **dataset_config)

    def _add_signature_columns_map_func(self, batch_dict):
        signature_columns = {

            "query_key_mm": [os.path.join(subset, video_id) for subset, video_id in zip(batch_dict['subset'], batch_dict['id'])]}
        return batch_dict | signature_columns


    def _load_hf_dataset(self):
        subsets = []
        for subset_name in subset_names:
            dataset = load_dataset(DATASET_HF_PATH, subset_name, split="test")
            new_column = [subset_name] * len(dataset)
            dataset = dataset.add_column("subset", new_column)
            subsets.append(dataset)
        dataset = datasets.concatenate_datasets(subsets)
        dataset = dataset.map(self._add_signature_columns_map_func, batched=True, num_proc=4, load_from_cache_file=False)
        return dataset, None

    def _process_one_sample(self, data_idx, batch_dict, *args, **kwargs):
        model_backbone   = kwargs["model_backbone"]
        image_resolution = kwargs["image_resolution"]   # resolutions kept as None to mirror original
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]
        num_frames       = kwargs["num_frames"]

        # Fields
        video_id       = batch_dict["id"][data_idx]
        query_raw      = batch_dict["question"][data_idx]
        answer_idx_raw = batch_dict["answer"][data_idx]          # int index
        question_type  = batch_dict["question_type"][data_idx]
        options        = batch_dict["options"][data_idx]
        subset         = batch_dict["subset"][data_idx]
        _image_unused  = batch_dict["image"][data_idx]           # unused, kept for parity

        # If not multiple-choice, return an empty-candidate sample (so parent can choose to skip if desired)
        if question_type != "multiple-choice":
            return {
                "query_text": "",
                "query_image": None,
                "cand_text": [],
                "cand_image": [],
                "dataset_infos": {
                    "video_id": video_id,
                    "subset": subset,
                    "status": "skipped_non_mc",
                },
            }

        # Build query with tokens and MC template
        q_with_token = process_query(
            query_raw,
            prompt=TASK_INST_QRY,
            video_token=VLM_VIDEO_TOKENS[model_backbone],
        )
        query_text, _cands_ignored, _ans_text_unused, _ans_idx_unused = qa_template(
            q_with_token, options, answer_idx_raw
        )

        # Paths & frame extraction (if missing)
        video_path = f"{video_root}/{subset}/{video_id}.mp4"
        frame_dir  = f"{frame_root}/{subset}/{video_id}"

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

        # Candidates: text-only multiple choice; strip "X. " prefix if present
        cand_text  = [o[o.find(". "):].strip(". ") for o in options]
        cand_image = [None] * len(options)

        dataset_info = {
            "video_id": video_id,
            "query": query_text,
            "cand_names": options,
            "answer": options[answer_idx_raw],
            "label_name": answer_idx_raw,
            "answer_idx": answer_idx_raw,
            "qry_frame_paths": qry_frame_paths,
            "subset": subset,
        }

        return {
            "query_text": query_text,     # str
            "query_image": query_image,   # dict with paths/bytes/resolutions
            "cand_text": cand_text,       # list[str]
            "cand_image": cand_image,     # list[None]
            "dataset_infos": dataset_info,
        }