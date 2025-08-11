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


TASK_INST_QRY = "Find a video that includes the following described scenes:"
TASK_INST_TGT = "Understand the content of the provided video."

DATASET_PARSER_NAME = "didemo"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class DiDemoEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, processor, 
                         query_key_text="caption", query_key_mm = None,       
                         cand_key_text=None, cand_key_mm="video"
                         **dataset_config)

    def _load_hf_dataset(self):
        return load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_name]), None


    def _process_one_sample(self, data_idx, batch_dict, **kwargs):
        image_resolution = kwargs["image_resolution"]
        model_backbone   = kwargs["model_backbone"]
        num_frames       = kwargs["num_frames"]
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]

        # Pull this sample's fields
        video_rel = batch_dict["video"][data_idx]
        caption   = batch_dict["caption"][data_idx]

        video_basename = os.path.basename(video_rel)
        video_name     = os.path.splitext(video_basename)[0]
        video_path     = os.path.join(video_root, video_basename)
        frame_dir      = os.path.join(frame_root, video_name)

        # Defaults (parent handles chat-template)
        query_text  = process_input_text(TASK_INST_QRY, model_backbone, text=caption)
        query_image = None
        cand_text, cand_image = [], []
        dataset_infos = {"cand_names": [video_name], "label_name": video_name}

        try:
            # Extract & process frames
            save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
            frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

            if frame_paths:
                cand_text.append(process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True))
                cand_image.append(
                    ImageVideoInstance(
                        bytes=[None] * len(frame_paths),
                        paths=frame_paths,
                        resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(frame_paths),
                    ).to_dict()
                )
            else:
                dataset_infos["error"] = "No frames returned from process_video_frames."

        except Exception as e:
            dataset_infos["error"] = str(e)

        return {
            "query_text": query_text,
            "query_image": query_image,     # None => no query-side image/video
            "cand_text": cand_text,         # list[str]
            "cand_image": cand_image,       # list[dict], zipped with cand_text
            "dataset_infos": dataset_infos, # per-sample info
        }

 
    # @add_metainfo_hook
    # def batch_preprocess(self, batch_dict, **kwargs):
    #     image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    #     num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    #     video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
    #     model_backbone = kwargs['model_backbone']

    #     query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    #     for video_path, caption in zip(batch_dict['video'], batch_dict['caption']):
            
    #         query_images.append([None])

    #         video_name = os.path.splitext(os.path.basename(video_path))[0]
    #         video_path = os.path.join(video_root, os.path.basename(video_path))
    #         frame_dir = os.path.join(frame_root, video_name)
    #         save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
    #         video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

    #         if self.apply_chat_template:
    #             query_texts.append([self.format_text_for_chat_template(
    #                 process_input_text(TASK_INST_QRY, model_backbone, text=caption), 
    #                 description=self.query_descriptions[(caption, video_path)] if self.query_descriptions is not None else None,
    #                 add_generation_prompt=self.model_args.do_sft_query)])
    #             cand_texts.append([self.prepared_targets[("", video_name)]])

    #         else:
    #             query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=caption)])
    #             cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)])
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


# DATASET_PARSER_NAME = "didemo"
# @AutoEvalPairDataset.register(DATASET_PARSER_NAME)
# def load_didemo_dataset(model_args, data_args, **kwargs):
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
