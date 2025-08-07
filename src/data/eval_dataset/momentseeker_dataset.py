import os

from datasets import load_dataset
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset

from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, MMEBV2EvalDatasetProcessor
from src.data.eval_dataset.base_eval_dataset import ImageVideoInstance
from src.data.utils.vision_utils import sample_frames, load_frames, VID_EXTENSIONS, save_frames
from src.model.processor import process_input_text
from ..prompts import (get_query, get_target, 
                       IMAGE_TASKS, VIDEO_TASKS, VISDOC_TASKS,
                       format_description, format_text_for_chat_template, 
                       extract_query_from_mmeb, extract_target_from_mmeb)

TASK_INST_QRY_TEXT = "Find the clip that corresponds to the given text:"
TASK_INST_QRY_IMG = "Select the video clip that aligns with the given text and image:"
TASK_INST_QRY_VIDEO = "Find the clip that corresponds to the given sentence and video segment:"
TASK_INST_TGT = "Understand the content of the provided video clip."

DATASET_PARSER_NAME = "momentseeker"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class MomentSeekerEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, processor, 
                         query_instruction=TASK_INST_QRY_TEXT, target_instruction=TASK_INST_TGT, target_modality="video",
                         **dataset_config)

    def _add_signature_columns_map_func(self, batch_dict):
        signature_columns = {

            # @xuanming we assume modality in the order of text, image, video
            # current assume two modalities max for query and target
            "query_key_text": batch_dict['query'],
            "query_key_mm": batch_dict['input_frames'],
            "cand_key_text": [""] * len(batch_dict['query']),
            "cand_key_mm": batch_dict['positive_frames']}
        return batch_dict | signature_columns


    def prepare_targets(self):
        """
            Precompute targets to avoid repetitive processing 1000x for each sample.
        """

        unique_pairs = set()
        for row in self.dataset:
            assert len(row["cand_key_text"]) == len(row["cand_key_mm"])
            for cand_text, cand_mm in zip(row["cand_key_text"], row["cand_key_mm"]):
                unique_pairs.add((cand_text, cand_mm))
            for cand_text, cand_mm in zip(row["query_key_text"], row['negative_frames']):
                unique_pairs.add((cand_text, cand_mm))

        preprocessed_pairs = {}

        for cand_text, cand_mm in unique_pairs:

            description = self.target_descriptions[(cand_text, cand_mm)] if self.target_descriptions is not None else None

            cand_text_processed = process_input_text(self.target_instruction, self.model_backbone, text=cand_text)
            input_kwargs = {
                "text": cand_text_processed,
                "description": description,
                "add_generation_prompt": self.model_args.do_sft_target
            }

            if self.target_modality == "video":
                input_kwargs["video_path"] = cand_mm
            elif self.target_modality == "text":
                input_kwargs["image_path"] = cand_mm

            preprocessed_pairs[(cand_text, cand_mm)] = self.format_text_for_chat_template(**input_kwargs)
        
        return preprocessed_pairs

    
    def _load_hf_dataset(self):
        if self.dataset_config.get("data_path", None) != None:
            dataset = load_dataset("json", data_files=self.dataset_config["data_path"])
            dataset = dataset["train"]
        else:
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']])

        return dataset, None

    @add_metainfo_hook
    def batch_process(self, batch_dict, *args, **kwargs):
        image_resolution = kwargs['image_resolution']
        ## metadata
        num_negative_clips = kwargs["num_negative_clips"]
        num_video_frames = kwargs["num_video_frames"]
        model_backbone = kwargs["model_backbone"]
        video_root, clip_root, frame_root = kwargs["video_root"], kwargs["clip_root"], kwargs["frame_root"]

        query_texts, query_images, cand_texts, cand_clip_images, dataset_infos = [], [], [], [], []
        for query, positive_frames, negative_frames, input_frames in \
                zip(batch_dict['query'], batch_dict["positive_frames"], batch_dict["negative_frames"], batch_dict["input_frames"]):

            if (input_frames.endswith(".mp4")):
                if self.apply_chat_template:
                    query_texts.append([self.format_text_for_chat_template(
                        process_input_text(TASK_INST_QRY_VIDEO, model_backbone, text=query), 
                        description=self.query_description[(query, input_frames)], 
                        video_path=input_frames, 
                        add_generation_prompt=self.model_args.do_sft_query)])
                else:
                    query_texts.append([process_input_text(TASK_INST_QRY_VIDEO, model_backbone, text=query, add_video_token=True)])
                query_video_name = input_frames.split(".mp4")[0].replace("/", "_")
                if query_video_name == 'movie101_77':  # TODO @yuepeng a buggy video?
                    pass
                query_frame_dir = os.path.join(frame_root, "video_frames", query_video_name)
                if not os.path.exists(query_frame_dir):
                    query_video_path = os.path.join(video_root, input_frames)
                    save_frames(video_path=query_video_path,
                                frame_dir=query_frame_dir,
                                max_frames_saved=num_video_frames)
                qry_frame_paths = load_frames(query_frame_dir)
                query_images.append([ImageVideoInstance(
                    bytes=[None] * len(qry_frame_paths),
                    paths=qry_frame_paths,
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_frame_paths),
                ).to_dict()])
            elif (input_frames.endswith(".jpg")):

                if self.apply_chat_template:
                    query_texts.append([self.format_text_for_chat_template(
                        process_input_text(TASK_INST_QRY_IMG, model_backbone, text=query), image_path=input_frames, description=self.query_description[(query, input_frames)], add_generation_prompt=self.model_args.do_sft_query)])
                else:
                    query_texts.append([process_input_text(TASK_INST_QRY_IMG, model_backbone, text=query, add_image_token=True)])
                input_image_path = os.path.join(frame_root, "", f"query_{input_frames}")
                query_images.append([ImageVideoInstance(
                    bytes=[None],
                    paths=[input_image_path],
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
                ).to_dict()])
            else:
                if self.apply_chat_template:
                    query_texts.append([self.format_text_for_chat_template(
                        process_input_text(TASK_INST_QRY_TEXT, model_backbone, text=query), add_generation_prompt=self.model_args.do_sft_query)])
                else:
                    query_texts.append([process_input_text(TASK_INST_QRY_TEXT, model_backbone, text=query)])
                query_images.append([None])

            pos_clip_paths = [entry["output_path"] for entry in positive_frames]
            neg_clip_paths = [entry["output_path"] for entry in negative_frames]

            pos_clip_name, cand_clip_names, cand_frames = [], [], []
            for path in pos_clip_paths:
                cand_clip_name = path.replace("/", "_").split(".mp4")[0]
                cand_clip_frame_dir = os.path.join(frame_root, "video_frames", cand_clip_name)
                if not os.path.exists(cand_clip_frame_dir):
                    cand_clip_abs_path = os.path.join(clip_root, path)
                    save_frames(video_path=cand_clip_abs_path, frame_dir=cand_clip_frame_dir, max_frames_saved=num_video_frames)
                pos_clip_frames = load_frames(cand_clip_frame_dir)
                cand_frames.append(ImageVideoInstance(
                    bytes=[None] * len(pos_clip_frames),
                    paths=pos_clip_frames,
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(pos_clip_frames),
                ).to_dict())
                cand_clip_names.append(cand_clip_frame_dir)
                pos_clip_name.append(cand_clip_frame_dir)
            for path in neg_clip_paths:
                cand_clip_name = path.replace("/", "_").split(".mp4")[0]
                cand_clip_frame_dir = os.path.join(frame_root, "video_frames", cand_clip_name)
                if not os.path.exists(cand_clip_frame_dir):
                    cand_clip_abs_path = os.path.join(clip_root, path)
                    save_frames(video_path=cand_clip_abs_path, frame_dir=cand_clip_frame_dir, max_frames_saved=num_video_frames)
                neg_clip_frames = load_frames(cand_clip_frame_dir)
                cand_frames.append(ImageVideoInstance(
                    bytes=[None] * len(neg_clip_frames),
                    paths=neg_clip_frames,
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(neg_clip_frames),
                ).to_dict())
                cand_clip_names.append(cand_clip_frame_dir)
            cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)] * len(pos_clip_paths + neg_clip_paths))
            cand_clip_images.append(cand_frames)
            dataset_infos.append({
                "cand_names": cand_clip_names,
                "label_name": pos_clip_name,
            })

        processed_batch = {"query_text": query_texts, "query_image": query_images,
                "cand_text": cand_texts, "cand_image": cand_clip_images,
                "dataset_infos": dataset_infos}
        return batch_dict | processed_batch


# DATASET_PARSER_NAME = "momentseeker"
# @AutoEvalPairDataset.register(DATASET_PARSER_NAME)
# def load_momentseeker_dataset(model_args, data_args, *args, **kwargs):
#     if kwargs.get("data_path", None) != None:
#         dataset = load_dataset("json", data_files=kwargs["data_path"])
#         dataset = dataset["train"]
#     else:
#         dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
#     dataset = sample_dataset(dataset, **kwargs)

#     kwargs['model_backbone'] = model_args.model_backbone
#     kwargs['image_resolution'] = data_args.image_resolution

#     dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
#                           batch_size=2048, num_proc=1,
#                           drop_last_batch=False, load_from_cache_file=False)
#     dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
#     return dataset, None
