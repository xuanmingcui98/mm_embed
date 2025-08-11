import os

from datasets import load_dataset
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames, VID_EXTENSIONS
from src.model.processor import process_input_text
from ..prompts import (get_query, get_target, 
                       IMAGE_TASKS, VIDEO_TASKS, VISDOC_TASKS,
                       format_description, format_text_for_chat_template, 
                       extract_query, extract_target)

TASK_INST_QRY = "Find the clip that corresponds to the described scene in the given video:"
TASK_INST_TGT = "Understand the content of the provided video."

DATASET_PARSER_NAME = "moment_retrieval"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class MomentRetrievalEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, processor, 
                        query_instruction=TASK_INST_QRY, target_instruction=TASK_INST_TGT, target_modality="video", **dataset_config)

    def _add_signature_columns_map_func(self, batch_dict):
        signature_columns = {

            # @xuanming we assume modality in the order of text, image, video
            # current assume two modalities max for query and target
            "query_key_text": batch_dict['query'],
            "query_key_mm": batch_dict['video_path'],
            "cand_key_text": [""] * len(batch_dict['query']),
            "cand_key_mm": batch_dict['video_path']}
        return batch_dict | signature_columns

    # def prepare_targets(self):
    #     """
    #         Precompute targets to avoid repetitive processing 1000x for each sample.
    #     """

    #     unique_pairs = set()
    #     for row in self.dataset:
    #         assert len(row["cand_key_text"]) == len(row["cand_key_mm"])
    #         for cand_text, cand_image in zip(row["cand_key_text"], row["cand_key_mm"]):
    #             unique_pairs.add((cand_text, cand_image))

    #     preprocessed_pairs = {}

    #     for cand_text, cand_image in unique_pairs:

    #         description = None
    #         if self.target_descriptions is not None:
    #             description = format_description(self.target_descriptions[(cand_text, cand_image)], self.data_args.use_cot)

    #         cand_text_processed = process_input_text(TASK_INST_TGT, self.model_backbone, text=cand_text)
    #         cand_text_processed = format_text_for_chat_template(self.processor, cand_text_processed, cand_image, description=description, add_generation_prompt=self.model_args.do_sft_target) + self.meta_queries

    #         preprocessed_pairs[(cand_text, cand_image)] = cand_text_processed
        
    #     return preprocessed_pairs

    @add_metainfo_hook
    def batch_preprocess(self, batch_dict, *args, **kwargs):
        image_resolution = kwargs['image_resolution']
        max_video_frames_saved = kwargs["max_video_frames_saved"]
        max_clip_frames_saved = kwargs["max_clip_frames_saved"]
        num_video_frames = kwargs["num_video_frames"]
        num_clip_frames = kwargs["num_clip_frames"]
        model_backbone = kwargs["model_backbone"]
        video_root, clip_root, frame_root = kwargs["video_root"], kwargs["clip_root"], kwargs["frame_root"]

        query_texts, query_images, cand_texts, cand_clip_images, dataset_infos = [], [], [], [], []

        for query, query_video_path in zip(batch_dict['query'], batch_dict['video_path']):

            video_name = os.path.splitext(os.path.basename(query_video_path))[0]
            frames_dir = os.path.join(frame_root, video_name)

            # Load query video
            query_video_path = os.path.join(video_root, os.path.basename(query_video_path)) if video_root else None
            query_frame_dir = os.path.join(frames_dir, "query")
            if not os.path.exists(query_frame_dir):
                save_frames(video_path=query_video_path,
                            frame_dir=query_frame_dir,
                            max_frames_saved=max_video_frames_saved)
            qry_frame_paths = process_video_frames(query_frame_dir, num_frames=num_video_frames)

            query_images.append([ImageVideoInstance(
                bytes=[None] * len(qry_frame_paths),
                paths=qry_frame_paths,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_frame_paths),
            ).to_dict()])

            # Load pos and neg clip, save the frames if only the raw video is provided.
            if not os.path.exists(frames_dir):
                clip_video_dir = os.path.join(clip_root, video_name) if clip_root else None
                clip_video_paths = [f for f in os.listdir(clip_video_dir) if os.path.splitext(f)[1].lower() in VID_EXTENSIONS]
                for clip_video_path in clip_video_paths:
                    clip_name = os.path.splitext(clip_video_path)[0]
                    clip_frame_dir_or_file = os.path.join(frames_dir, clip_name)
                    clip_video_path_abs = os.path.join(clip_video_dir, clip_video_path)
                    save_frames(video_path=clip_video_path_abs,
                                frame_dir=clip_frame_dir_or_file,
                                max_frames_saved=max_clip_frames_saved)
            cand_clip_names, cand_frames = [], []
            for clip_frame_dir_or_file in os.listdir(frames_dir):
                clip_frame_dir_abs = os.path.join(frames_dir, clip_frame_dir_or_file)
                if clip_frame_dir_or_file == 'query' or os.path.isfile(clip_frame_dir_abs):
                    continue
                if clip_frame_dir_or_file.startswith("positive"):
                    pos_clip_name = clip_frame_dir_abs
                cand_frame_paths = process_video_frames(clip_frame_dir_abs, num_frames=num_clip_frames)
                cand_frames.append(ImageVideoInstance(
                    bytes=[None] * len(cand_frame_paths),
                    paths=cand_frame_paths,
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(cand_frame_paths),
                ).to_dict())
                cand_clip_names.append(clip_frame_dir_abs)  # use absolute path here instead of file name to keep it unique

            if self.apply_chat_template:
                query_texts.append([self.format_text_for_chat_template(
                    process_input_text(TASK_INST_QRY, model_backbone, text=query), 
                    description=self.query_descriptions[(query, video_name)] if self.query_descriptions is not None else None,
                    video_path=query_frame_dir)])
                cand_texts.append([self.format_text_for_chat_template(
                    process_input_text(TASK_INST_TGT, model_backbone), 

                    video_path=frames_dir)] * len(cand_clip_names))
            else:
                query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=query, add_video_token=True)])
                cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)] * len(cand_clip_names))
            cand_clip_images.append(cand_frames)
            dataset_infos.append({
                "cand_names": cand_clip_names,
                "label_name": pos_clip_name,
            })

        processed_batch = {"query_text": query_texts, "query_image": query_images,
                "cand_text": cand_texts, "cand_image": cand_clip_images,
                "dataset_infos": dataset_infos}
        return batch_dict | processed_batch



# def load_moment_retrieval_dataset(model_args, data_args, **kwargs):
#     if kwargs.get("data_path", None) != None:
#         dataset = load_dataset("json", data_files=kwargs["data_path"])
#         dataset = dataset["train"]
#     else:
#         dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
#     dataset = sample_dataset(dataset, **kwargs)

#     kwargs['model_backbone'] = model_args.model_backbone
#     kwargs['image_resolution'] = data_args.image_resolution
    
#     dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
#                           batch_size=2048, num_proc=8,
#                           drop_last_batch=False, load_from_cache_file=False)
#     dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

#     return dataset, None
