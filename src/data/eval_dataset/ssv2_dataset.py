import os

from datasets import Dataset
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.eval_dataset.video_classification_utils import VIDEOCLS_LABEL_MAPPING, DATASET_INSTRUCTION
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text
from ..prompts import (get_query, get_target, 
                       IMAGE_TASKS, VIDEO_TASKS, VISDOC_TASKS,
                       format_description, format_text_for_chat_template, 
                       extract_query_from_mmeb, extract_target_from_mmeb)

TASK_INST_TGT = "Represent the following text answer to a question.\nAnswer: "

DATASET_PARSER_NAME = "ssv2"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class SSV2EvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, processor, 
                         query_instruction=DATASET_INSTRUCTION[dataset_config['dataset_name']],
                         target_instruction=TASK_INST_TGT,
                         target_modality="text", 
                         **dataset_config)

        self.dataset_config['image_resolution'] = data_args.image_resolution

    def _load_hf_dataset(self):
        return load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_name]), None

    def add_signature_columns(self):

        self.dataset = self.dataset.add_column("cand_key_text", self.dataset["pos_text"])
        self.dataset = self.dataset.add_column("cand_key_mm", [""] * len(self.dataset))


    def _add_signature_columns_map_func(self, batch_dict):
        signature_columns = {

            # @xuanming we assume modality in the order of text, image, video
            # current assume two modalities max for query and target
            "query_key_text": [""] * len(batch_dict['video_id']),
            "query_key_mm": batch_dict['video_id'],
            "cand_key_text": batch_dict['pos_text'],
            "cand_key_mm": [""] * len(batch_dict['question'])}
        return batch_dict | signature_columns

    @add_metainfo_hook
    def batch_preprocess(self, batch_dict, **kwargs):
        image_resolution = kwargs['image_resolution']
        num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
        video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
        dataset_name = kwargs['dataset_name']
        model_backbone = kwargs['model_backbone']

        query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
        for video_id, pos_text, cand_text in \
            zip(batch_dict['video_id'], batch_dict['pos_text'], batch_dict['neg_text']):

            video_path = os.path.join(video_root, str(video_id) + '.mp4')
            frame_dir = os.path.join(frame_root, str(video_id))
            save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
            video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

            if self.apply_chat_template:
                query_texts.append([format_text_for_chat_template(
                    self.processor, process_input_text(DATASET_INSTRUCTION[dataset_name], model_backbone), video_path=frame_dir)])
                cand_texts.append(self.prepared_targets[("", pos_text)])
            else:
                query_texts.append([process_input_text(DATASET_INSTRUCTION[dataset_name], model_backbone, add_video_token=True)])
                cand_texts.append(cand_text)

            query_images.append([ImageVideoInstance(
                bytes=[None] * len(video_frame_paths),
                paths=video_frame_paths,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
            ).to_dict()])

            cand_images.append([None] * len(cand_text))
            dataset_info = {
                "cand_names": cand_text,
                "label_name": pos_text,
            }
            dataset_infos.append(dataset_info)

        processed_batch = {"query_text": query_texts, "query_image": query_images,
                "cand_text": cand_texts, "cand_image": cand_images,
                "dataset_infos": dataset_infos}
        return batch_dict | processed_batch
    
# def load_ssv2_dataset(model_args, data_args, **kwargs):
#     """
#     ssv2-mc setup for zero-shot evaluation.
#     """
#     dataset_name = kwargs['dataset_name']
#     dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
#     dataset = sample_dataset(dataset, **kwargs)

#     kwargs['model_backbone'] = model_args.model_backbone
#     kwargs['image_resolution'] = data_args.image_resolution

#     dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
#                           batch_size=256, num_proc=4,
#                           drop_last_batch=False, load_from_cache_file=False)
#     dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
#     corpus = None

#     return dataset, corpus
