import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text


TASK_INST_QRY = "Find the video snippet that corresponds to the given summary:"
TASK_INST_TGT = "Understand the content of the provided video."

DATASET_PARSER_NAME = "msvd"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class MSVDEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
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
            "query_key_text": batch_dict['caption'],
            "query_key_mm": [""] * len(batch_dict['caption']),
            "cand_key_text": [""] * len(batch_dict['caption']),
            "cand_key_mm": batch_dict['video_id']}
        return batch_dict | signature_columns
    
    def _load_hf_dataset(self):
        return load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']]), None

    @add_metainfo_hook
    def batch_preprocess(self, batch_dict, **kwargs):
        image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
        num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
        video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
        model_backbone = kwargs['model_backbone']

        query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
        for video_name, video_path, captions in zip(batch_dict['video_id'], batch_dict["video"], batch_dict['caption']):

            query_images.append([None])

            video_path = os.path.join(video_root, video_path)
            frame_dir = os.path.join(frame_root, video_name)
            save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
            video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

            if self.apply_chat_template:
                query_texts.append([self.format_text_for_chat_template(
                    process_input_text(TASK_INST_QRY, model_backbone, text=captions[0]), video_path=frame_dir, add_generation_prompt=self.model_args.do_sft_query)])
                cand_texts.append([self.prepared_targets[("", video_name)]])
            else:
                query_texts.append([process_input_text(TASK_INST_QRY, model_backbone)])
                cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)])
            cand_images.append([ImageVideoInstance(
                bytes=[None] * num_frames,
                paths=video_frame_paths,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames,
            ).to_dict()])
            dataset_infos.append({
                "cand_names": [video_name],
                "label_name": video_name,
            })

        processed_batch = {"query_text": query_texts, "query_image": query_images,
                "cand_text": cand_texts, "cand_image": cand_images,
                "dataset_infos": dataset_infos}
        return batch_dict | processed_batch



# DATASET_PARSER_NAME = "msvd"
# @AutoEvalPairDataset.register(DATASET_PARSER_NAME)
# def load_msvd_dataset(model_args, data_args, **kwargs):
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
