import os
import sys

from datasets import load_dataset

from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import sample_dataset
from src.data.utils.vision_utils import temporal_random_crop, process_video_frames, load_frames
from src.model.processor import VLM_VIDEO_TOKENS
import torchvision
import cv2

def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token} {prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query


TASK_INST_QRY = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: "
TASK_INST_TGT = "Represent the following text:\n"

OPTIONS = ['A', 'B', 'C', 'D']


DATASET_PARSER_NAME = "videomme"
DATASET_HF_PATH = "lmms-lab/Video-MME"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class VideoMMMEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, processor, 
                         query_key_text='question', query_key_mm='videoID', cand_key_text=None, cand_key_mm='options',
                         **dataset_config)

    def _add_signature_columns_map_func(self, batch_dict):
        signature_columns = {

            # @xuanming we assume modality in the order of text, image, video
            # current assume two modalities max for query and target
            "query_key_text": batch_dict['question'],
            "query_key_mm": batch_dict['videoID'],
            "cand_key_text": [""] * len(batch_dict['question']),
            "cand_key_mm": [""] * len(batch_dict['question'])}
        return batch_dict | signature_columns

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

        # Build query text with options included
        query_text = process_query(
            query + "\n" + "\n".join(options),
            prompt=TASK_INST_QRY,
            video_token=VLM_VIDEO_TOKENS[model_backbone],
        )

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
        }


    # @add_metainfo_hook
    # def batch_process(self, batch_dict, *args, **kwargs):
    #     model_backbone = kwargs['model_backbone']
    #     image_resolution = kwargs['image_resolution']
    #     max_frames_saved = kwargs['max_frames_saved']
    #     video_root = kwargs['video_root']
    #     frame_root = kwargs['frame_root']
    #     num_frames = kwargs['num_frames']
    #     query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    #     batch_size = len(batch_dict['question']) if batch_dict['question'] else 0
    #     for query, video_id, options, answer, question_id, domain, sub_category in (
    #             zip(batch_dict['question'], batch_dict['videoID'], batch_dict['options'], batch_dict['answer'], batch_dict['question_id'], batch_dict['domain'], batch_dict['sub_category'])):
    #         query = process_query(query + '\n' + '\n'.join(options), prompt=self.query_instruction, video_token=VLM_VIDEO_TOKENS[model_backbone])
    #         query_texts.append([query])
    #         video_path = f'{video_root}/{video_id}.mp4'
    #         frame_dir = f'{frame_root}/{video_id}'
    #         frames = load_frames(frame_dir)
    #         if not frames:
    #             print(f'Extracting frames for: {video_path}')
    #             os.makedirs(frame_dir, exist_ok=True)
    #             assert os.path.exists(video_path)
    #             cap = cv2.VideoCapture(video_path)
    #             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             step = max(1, total_frames // max_frames_saved)
    #             frame_idx = 0
    #             saved_frames = 0
    #             while saved_frames < max_frames_saved:
    #                 assert cap.isOpened(), "not cap.isOpened()"
    #                 cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Move to specific frame
    #                 ret, frame = cap.read()
    #                 if not ret:
    #                     break
    #                 frame_path = os.path.join(frame_dir, f"{saved_frames:04d}.jpeg")
    #                 cv2.imwrite(frame_path, frame)
    #                 saved_frames += 1
    #                 frame_idx += step
    #             cap.release()
    #             # print(f'Extracted #frames: {saved_frames}, dumped to {frame_dir}')

    #         qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
    #         # print(f'Loaded #frames: {len(qry_frame_paths)}, from {frame_dir}')
    #         qry_frames = {"bytes": [None] * len(qry_frame_paths), "paths": qry_frame_paths, "resolutions": [None] * len(qry_frame_paths)}
    #         query_images.append([qry_frames])
    #         cand_texts.append([o[o.find('. '):].strip('. ') for o in options])
    #         cand_images.append([None] * len(options))
    #         dataset_info = {
    #             "question_id": question_id,
    #             "video_id": video_id,
    #             "query": query,
    #             "cand_names": options,
    #             "answer": answer,
    #             "label_name": options[OPTIONS.index(answer)],
    #             "answer_idx": OPTIONS.index(answer),
    #             "domain": domain,
    #             "sub_category": sub_category,
    #             "qry_frame_paths": qry_frame_paths,
    #         }
    #         dataset_infos.append(dataset_info)
    #     if len(query_texts) == 0:
    #         print('something went wrong')
    #     # print_rank(f"dataset.map(): global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    #     processed_batch = {"query_text": query_texts, "query_image": query_images,
    #             "cand_text": cand_texts, "cand_image": cand_images,
    #             "dataset_infos": dataset_infos}
    #     return batch_dict | processed_batch


# def load_videomme_dataset(model_args, data_args, *args, **kwargs):
#     dataset = load_dataset(DATASET_HF_PATH, split="test")
#     print(f"Loading {DATASET_HF_PATH}, {len(dataset)} samples")
#     kwargs['dataset_name'] = DATASET_PARSER_NAME
#     kwargs['model_backbone'] = model_args.model_backbone
#     kwargs['image_resolution'] = data_args.image_resolution
#     kwargs['global_dataset_name'] = DATASET_PARSER_NAME
#     dataset = sample_dataset(dataset, **kwargs)
#     dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
#                           batch_size=256, num_proc=4,
#                           drop_last_batch=False, load_from_cache_file=False)

#     return dataset, None
