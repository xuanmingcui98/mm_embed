import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text
from ..prompts import (get_query, get_target, 
                       IMAGE_TASKS, VIDEO_TASKS, VISDOC_TASKS,
                       format_description, format_text_for_chat_template, 
                       extract_query, extract_target)

TASK_INST_QRY = "Find a video that contains the following visual content:"
TASK_INST_TGT = "Understand the content of the provided video."

DATASET_PARSER_NAME = "msrvtt"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class MSRVTTEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, processor, 
                         query_key_text='caption', query_key_mm=None,
                         cand_key_text=None, cand_key_mm='video_id',
                         **dataset_config)

    def _load_hf_dataset(self):
        return load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']]), None

    def _process_one_sample(self, data_idx, batch_dict, **kwargs):
        image_resolution = kwargs["image_resolution"]
        max_frames_saved = kwargs["max_frames_saved"]
        num_frames       = kwargs["num_frames"]
        model_backbone   = kwargs["model_backbone"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]

        # Fields for this sample
        video_name = batch_dict["video_id"][data_idx]
        rel_video_path = batch_dict["video"][data_idx]
        caption = batch_dict["caption"][data_idx]

        # Paths
        abs_video_path = os.path.join(video_root, rel_video_path)
        frame_dir = os.path.join(frame_root, video_name)

        # Ensure frames exist and sample them
        save_frames(video_path=abs_video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Query: text only (no query-side image/video)
        query_text  = process_input_text(TASK_INST_QRY, model_backbone, text=caption)
        query_image = None

        # Candidate: video frames as image dict + generic target text
        cand_text = [process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)]
        cand_image = [{
            "bytes": [None] * len(frame_paths),
            "paths": frame_paths,
            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(frame_paths),
        }]

        dataset_info = {
            "cand_names": [video_name],
            "label_name": video_name,
        }

        return {
            "query_text": query_text,
            "query_image": query_image,
            "cand_text": cand_text,
            "cand_image": cand_image,
            "dataset_infos": dataset_info,
        }

    # @add_metainfo_hook
    # def batch_preprocess(self, batch_dict, **kwargs):
    #     image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    #     num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    #     video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
    #     model_backbone = kwargs['model_backbone']

    #     query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    #     for video_name, video_path, caption in (
    #             zip(batch_dict['video_id'], batch_dict['video'], batch_dict['caption'])):


    #         video_path = os.path.join(video_root, video_path)
    #         frame_dir = os.path.join(frame_root, video_name)
    #         save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
    #         video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

    #         if self.apply_chat_template:
    #             query_texts.append([self.format_text_for_chat_template(
    #                 process_input_text(TASK_INST_QRY, model_backbone, text=caption), video_path=frame_dir, add_generation_prompt=self.model_args.do_sft_query)])
    #             cand_texts.append([self.prepared_targets[("", video_name)]])
    #         else:
    #             query_texts.append([process_input_text(TASK_INST_QRY, model_backbone)])
    #             cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)])
    #         query_images.append([None])
    #         cand_images.append([ImageVideoInstance(
    #             bytes=[None] * num_frames,
    #             paths=video_frame_paths,
    #             resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames,
    #         ).to_dict()])
    #         dataset_infos.append({
    #             "cand_names": [video_name],
    #             "label_name": video_name,
    #         })

    #     processed_batch = {"query_text": query_texts, "query_image": query_images,
    #             "cand_text": cand_texts, "cand_image": cand_images,
    #             "dataset_infos": dataset_infos}
    #     return batch_dict | processed_batch


# DATASET_PARSER_NAME = "msrvtt"
# @AutoEvalPairDataset.register(DATASET_PARSER_NAME)
# def load_msrvtt_dataset(model_args, data_args, **kwargs):
#     dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
#     dataset = sample_dataset(dataset, **kwargs)

#     kwargs['model_backbone'] = model_args.model_backbone
#     kwargs['image_resolution'] = data_args.image_resolution

#     dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
#                           batch_size=256, num_proc=4,
#                           drop_last_batch=False, load_from_cache_file=False)
#     dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
#     corpus = None  # No additional corpus

#     return dataset, corpus
