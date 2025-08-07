import os
import sys

from datasets import load_dataset

from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import sample_dataset
from src.data.utils.vision_utils import temporal_random_crop, process_video_frames, load_frames, qa_template
from src.model.processor import VLM_VIDEO_TOKENS
import datasets
import cv2

def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token} {prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query


TASK_INST_QRY = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: "
TASK_INST_TGT = "Represent the following text answer to a question.\nAnswer: "
OPTIONS = ['A', 'B', 'C', 'D']

subset_names = ['Perception', 'Comprehension', 'Adaptation']

DATASET_PARSER_NAME = "videommmu"
DATASET_HF_PATH = "lmms-lab/VideoMMMU"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class VideoMMMUEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, processor, 
                         query_instruction=TASK_INST_QRY, target_instruction=TASK_INST_TGT, target_modality="text",
                         **dataset_config)

    def _add_signature_columns_map_func(self, batch_dict):
        signature_columns = {

            # @xuanming we assume modality in the order of text, image, video
            # current assume two modalities max for query and target
            "query_key_text": batch_dict['question'],
            "query_key_mm": [os.path.join(subset, video_id) for subset, video_id in zip(batch_dict['subset'], batch_dict['id'])],
            "cand_key_text": [""] * len(batch_dict['question']),
            "cand_key_mm": [""] * len(batch_dict['question'])}
        return batch_dict | signature_columns


    def _load_hf_dataset(self):
        subsets = []
        for subset_name in subset_names:
            dataset = load_dataset(DATASET_HF_PATH, subset_name, split="test")
            new_column = [subset_name] * len(dataset)
            dataset = dataset.add_column("subset", new_column)
            subsets.append(dataset)
        dataset = datasets.concatenate_datasets(subsets)
        return dataset, None

    @add_metainfo_hook
    def batch_preprocess(self, batch_dict, *args, **kwargs):
        model_backbone = kwargs['model_backbone']
        image_resolution = kwargs['image_resolution']
        max_frames_saved = kwargs['max_frames_saved']
        video_root = kwargs['video_root']
        frame_root = kwargs['frame_root']
        num_frames = kwargs['num_frames']
        query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
        batch_size = len(batch_dict['question']) if batch_dict['question'] else 0
        for video_id, query, answer, question_type, options, subset, image in \
                zip(batch_dict['id'], batch_dict['question'], batch_dict['answer'], batch_dict['question_type'], batch_dict['options'], batch_dict['subset'], batch_dict['image']):
            if question_type != 'multiple-choice':
                continue
            query = process_query(query, prompt=TASK_INST_QRY, video_token=VLM_VIDEO_TOKENS[model_backbone])
            query, cands, _, _ = qa_template(query, options, answer)
            query_texts.append([query])
            video_path = f'{video_root}/{subset}/{video_id}.mp4'
            frame_dir = f'{frame_root}/{subset}/{video_id}'
            frames = load_frames(frame_dir)
            if not frames:
                print(f'Extracting frames for: {video_path}')
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
                print(f'[{DATASET_PARSER_NAME}] Extracted #frames: {saved_frames}, dumped to {frame_dir}')

            qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
            # print(f'[{DATASET_PARSER_NAME}] Loaded #frames: {len(qry_frame_paths)}, from {frame_dir}')
            qry_frames = {"bytes": [None] * len(qry_frame_paths), "paths": qry_frame_paths, "resolutions": [None] * len(qry_frame_paths)}
            query_images.append([qry_frames])
            cand_texts.append([o[o.find('. '):].strip('. ') for o in options])
            cand_images.append([None] * len(options))
            dataset_info = {
                "video_id": video_id,
                "query": query,
                "cand_names": options,
                "answer": options[answer],
                "label_name": answer,
                "answer_idx": answer,
                "qry_frame_paths": qry_frame_paths,
            }
            dataset_infos.append(dataset_info)
        if len(query_texts) == 0:
            print('something went wrong')
        # print_rank(f"dataset.map(): global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
        processed_batch = {"query_text": query_texts, "query_image": query_images,
                "cand_text": cand_texts, "cand_image": cand_images,
                "dataset_infos": dataset_infos}
        return batch_dict | processed_batch



# def load_videommmu_dataset(model_args, data_args, training_args, *args, **kwargs):
#     subsets = []
#     for subset_name in subset_names:
#         dataset = load_dataset(DATASET_HF_PATH, subset_name, split="test")
#         new_column = [subset_name] * len(dataset)
#         dataset = dataset.add_column("subset", new_column)
#         subsets.append(dataset)
#     dataset = datasets.concatenate_datasets(subsets)
#     # TODO filter the dataset by question_type=='multiple-choice'

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

