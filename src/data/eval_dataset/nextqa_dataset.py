import os
import sys

from datasets import load_dataset

from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import sample_dataset
from src.data.utils.vision_utils import temporal_random_crop, process_video_frames, load_frames, qa_template
from src.model.processor import VLM_VIDEO_TOKENS
import torchvision
import cv2
from src.model.processor import process_input_text

def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token} {prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query


TASK_INST_QRY = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: "
TASK_INST_TGT = "Represent the following text answer to a question.\nAnswer: "

OPTIONS = ['A', 'B', 'C', 'D']


DATASET_PARSER_NAME = "nextqa"
DATASET_HF_PATH = "lmms-lab/NExTQA"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class NextQAEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, processor, 
                        query_instruction=TASK_INST_QRY, target_instruction=TASK_INST_TGT, target_modality="text", **dataset_config)


    def _add_signature_columns_map_func(self, batch_dict):
        signature_columns = {

            # @xuanming we assume modality in the order of text, image, video
            # current assume two modalities max for query and target
            "query_key_text": batch_dict['question'],
            "query_key_mm": batch_dict['video'],
            "cand_key_text": [[a0, a1, a2, a3, a4][answer] for answer, a0, a1, a2, a3, a4 in zip(batch_dict['answer'], batch_dict['a0'], batch_dict['a1'], batch_dict['a2'], batch_dict['a3'], batch_dict['a4'])],
            "cand_key_mm": [""] * len(batch_dict['question'])}
            
        return batch_dict | signature_columns


    def _load_hf_dataset(self):
        return load_dataset(DATASET_HF_PATH, "MC", split="test"), None

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
        for video_id, query, answer, qid, _type, a0, a1, a2, a3, a4 in \
                zip(batch_dict['video'], batch_dict['question'], batch_dict['answer'], batch_dict['qid'], batch_dict['type'], batch_dict['a0'], batch_dict['a1'], batch_dict['a2'], batch_dict['a3'], batch_dict['a4']):
            options = [a0, a1, a2, a3, a4]
            query = process_query(query, prompt=TASK_INST_QRY, video_token=VLM_VIDEO_TOKENS[model_backbone])
            query, cands, _, _ = qa_template(query, options, answer)

            video_path = f'{video_root}/{video_id}.mp4'
            frame_dir = f'{frame_root}/{video_id}'
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

            if self.apply_chat_template:
                query_texts.append([self.format_text_for_chat_template(query, video_path=frame_dir, add_generation_prompt=self.model_args.do_sft_query)])
                cand_texts.append([self.format_text_for_chat_template(process_input_text(TASK_INST_TGT, model_backbone, text=text), add_generation_prompt=self.model_args.do_sft_target) for text in cands])
            else:
                query_texts.append([query])
                cand_texts.append(options)
            cand_images.append([None] * len(options))
            dataset_info = {
                "question_id": qid,
                "video_id": video_id,
                "query": query,
                "cand_names": options,
                "answer": options[answer],
                "label_name": options[answer],
                "answer_idx": answer,
                "type": _type,
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



# DATASET_PARSER_NAME = "nextqa"
# DATASET_HF_PATH = "lmms-lab/NExTQA"
# @AutoEvalPairDataset.register(DATASET_PARSER_NAME)
# def load_nextqa_dataset(model_args, data_args, *args, **kwargs):
#     dataset = load_dataset(DATASET_HF_PATH, "MC", split="test")
#     print(f"Loading {DATASET_HF_PATH}, {len(dataset)} samples")

#     # dataset = dataset.filter(lambda example: example['video'] == 4740931975)
#     kwargs['dataset_name'] = DATASET_PARSER_NAME
#     kwargs['model_backbone'] = model_args.model_backbone
#     kwargs['image_resolution'] = data_args.image_resolution
#     kwargs['global_dataset_name'] = DATASET_PARSER_NAME
#     dataset = sample_dataset(dataset, **kwargs)
#     dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
#                           batch_size=256, num_proc=4,
#                           drop_last_batch=False, load_from_cache_file=False)

#     return dataset, None
