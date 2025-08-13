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
                       extract_query, extract_target)


TASK_INST_TGT = "Represent the following text:\n"

DATASET_PARSER_NAME = "video_classification"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class VideoClassificationEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__("video_classification", model_args, data_args, training_args, processor, 
                         query_key_text=None, query_key_mm='video_id',
                         cand_key_text='pos_text', cand_key_mm=None,
                         **dataset_config)

    def _load_hf_dataset(self):
        dataset_name = self.dataset_config['dataset_name']
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        corpus = Dataset.from_list([{
            "cand_text": [label],
            "cand_image": [None],
            "dataset_infos": {"cand_names": [label]}} for label in VIDEOCLS_LABEL_MAPPING[self.dataset_name]])
        return dataset, corpus

    def _process_one_sample(self, data_idx, batch_dict, *args, **kwargs):
        image_resolution = kwargs["image_resolution"]
        num_frames       = kwargs["num_frames"]
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]
        dataset_name     = kwargs["dataset_name"]
        model_backbone   = kwargs["model_backbone"]

        # Pull fields
        video_id = batch_dict["video_id"][data_idx]
        label    = batch_dict["pos_text"][data_idx]

        # Paths
        video_path = os.path.join(video_root, f"{video_id}.mp4")
        frame_dir  = os.path.join(frame_root, video_id)

        # Build query text from dataset instruction (parent will handle chat-template)
        query_text = process_input_text(
            DATASET_INSTRUCTION[dataset_name],
            model_backbone,
            add_video_token=True
        )

        query_image = None
        cand_text   = [label]
        cand_image  = [None]
        dataset_infos = {"cand_names": [label], "label_name": label}

        try:
            # Extract frames then sample/order them for the model
            save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
            frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

            if frame_paths:
                query_image = {
                    "bytes": [None] * len(frame_paths),
                    "paths": frame_paths,
                    "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(frame_paths),
                }
            else:
                dataset_infos["error"] = "No frames returned from process_video_frames."

        except Exception as e:
            dataset_infos["error"] = str(e)

        return {
            "query_text": query_text,   # str
            "query_image": query_image, # dict or None
            "cand_text": cand_text,     # [label]
            "cand_image": cand_image,   # [None]
            "dataset_infos": dataset_infos,
        }


    # def _add_signature_columns_map_func(self, batch_dict):
    #     signature_columns = {

    #         # @xuanming we assume modality in the order of text, image, video
    #         # current assume two modalities max for query and target
    #         "query_key_text": [""] * len(batch_dict['pos_text']),
    #         "query_key_mm": batch_dict['video_id'],
    #         "cand_key_text": [""] * len(batch_dict['pos_text']),
    #         "cand_key_mm": batch_dict['pos_text']}
    #     return batch_dict | signature_columns

    # @add_metainfo_hook
    # def batch_preprocess(self, batch_dict, **kwargs):
    #     image_resolution = self.dataset_config.get('image_resolution')
    #     num_frames, max_frames_saved = self.dataset_config['num_frames'], self.dataset_config['max_frames_saved']
    #     video_root, frame_root = self.dataset_config['video_root'], self.dataset_config['frame_root']
    #     dataset_name = self.dataset_config['dataset_name']
    #     model_backbone = self.dataset_config['model_backbone']

    #     query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    #     for video_id, label in zip(batch_dict['video_id'], batch_dict['pos_text']):
    #         video_path = os.path.join(video_root, video_id + '.mp4')
    #         frame_dir = os.path.join(frame_root, video_id)
    #         save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
    #         video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

    #         query_description = None
    #         if self.query_descriptions is not None:
    #             query_description = format_description(self.query_descriptions[("", video_id)])
    #         if self.data_args.apply_chat_template:
    #             query_texts.append([format_text_for_chat_template(self.processor, 
    #                                                               DATASET_INSTRUCTION[dataset_name], 
    #                                                               video_path=frame_dir, 
    #                                                               description=query_description)])
    #             cand_texts.append([self.prepared_targets[("", label)]])
    #         else:
    #             query_texts.append([process_input_text(DATASET_INSTRUCTION[dataset_name], model_backbone, add_video_token=True)])
    #             cand_texts.append([label])
    #         query_images.append([ImageVideoInstance(
    #             bytes=[None] * len(video_frame_paths),
    #             paths=video_frame_paths,
    #             resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
    #         ).to_dict()])
            
    #         cand_images.append([None])
    #         dataset_info = {
    #             "cand_names": [label],
    #             "label_name": label,
    #         }
    #         dataset_infos.append(dataset_info)


    #     # expand returned dict to include any fields in the original dataset
    #     processed_batch = {"query_text": query_texts, "query_image": query_images,
    #             "cand_text": cand_texts, "cand_image": cand_images,
    #             "dataset_infos": dataset_infos}

    #     processed_batch = batch_dict | processed_batch
    #     return processed_batch


    # @add_metainfo_hook
    # def batch_preprocess_bm(self, batch_dict, **kwargs):
    #     image_resolution = kwargs['image_resolution']
    #     num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    #     video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
    #     dataset_name = kwargs['dataset_name']
    #     model_backbone = kwargs['model_backbone']

    #     query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    #     for video_id, label in zip(batch_dict['video_id'], batch_dict['pos_text']):
    #         video_path = os.path.join(video_root, video_id + '.mp4')
    #         frame_dir = os.path.join(frame_root, video_id)
    #         save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
    #         video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

    #         query_texts.append([process_input_text(DATASET_INSTRUCTION[dataset_name], model_backbone, add_video_token=True)])
    #         query_images.append([ImageVideoInstance(
    #             bytes=[None] * len(video_frame_paths),
    #             paths=video_frame_paths,
    #             resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
    #         ).to_dict()])

    #         cand_texts.append([label])
    #         cand_images.append([None])
    #         dataset_info = {
    #             "cand_names": [label],
    #             "label_name": label,
    #         }
    #         dataset_infos.append(dataset_info)

    #     processed_batch = {"query_text": query_texts, "query_image": query_images,
    #             "cand_text": cand_texts, "cand_image": cand_images,
    #             "dataset_infos": dataset_infos}
    #     return batch_dict | processed_batch



# def load_video_class_dataset(model_args, data_args, **kwargs):
#     dataset_name = kwargs['dataset_name']
#     dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
#     dataset = sample_dataset(dataset, **kwargs)

#     kwargs['model_backbone'] = model_args.model_backbone
#     kwargs['image_resolution'] = data_args.image_resolution

#     dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
#                           batch_size=256, num_proc=4,
#                           drop_last_batch=False, load_from_cache_file=False)
#     dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
#     corpus = Dataset.from_list([{
#         "cand_text": [label],
#         "cand_image": [None],
#         "dataset_infos": {"cand_names": [label]}} for label in VIDEOCLS_LABEL_MAPPING[dataset_name]])

#     return dataset, corpus
