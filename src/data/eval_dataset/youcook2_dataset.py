import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text


TASK_INST_QRY = "Find a video that demonstrates the following action while making a recipe:"
TASK_INST_TGT = "Understand the content of the provided video."

DATASET_PARSER_NAME = "youcook2"
# slightly less than the official one: https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/csv/validation_youcook.csv?plain=1
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class YouCook2EvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__("youcook2", model_args, data_args, training_args, processor, 
                         query_key_text="sentence", query_key_mm=None, cand_key_text=None, cand_key_mm="id", **dataset_config)

    def _load_hf_dataset(self):
        return load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']]), None

    def _process_one_sample(self, data_idx, batch_dict, **kwargs):
        image_resolution = kwargs["image_resolution"]
        num_frames       = kwargs["num_frames"]
        max_frames_saved = kwargs["max_frames_saved"]
        video_root       = kwargs["video_root"]
        frame_root       = kwargs["frame_root"]
        model_backbone   = kwargs["model_backbone"]

        # Fields
        video_name      = batch_dict["id"][data_idx]
        rel_video_path  = batch_dict["video_path"][data_idx]
        text            = batch_dict["sentence"][data_idx]

        # Paths
        abs_video_path = os.path.join(video_root, os.path.basename(rel_video_path))
        frame_dir      = os.path.join(frame_root, video_name)

        # Ensure frames and sample
        save_frames(video_path=abs_video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Query: text only
        query_text  = process_input_text(TASK_INST_QRY, model_backbone, text=text)
        query_image = None

        # Candidate: video frames + target text
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


    # def _add_signature_columns_map_func(self, batch_dict):
    #     signature_columns = {
    #         "query_key_text": batch_dict['sentence'],
    #         "query_key_mm": [''] * len(batch_dict['sentence']),
    #         "cand_key_text": [''] * len(batch_dict['sentence']),
    #         "cand_key_mm": batch_dict['video_path']}
    #     return batch_dict | signature_columns

    # def add_signature_columns(self):
    #     self.dataset = self.dataset.add_column("cand_key_text", [""] * len(self.dataset))
    #     self.dataset = self.dataset.add_column("cand_key_mm", self.dataset['video_path'])

    # @add_metainfo_hook
    # def batch_preprocess(self, batch_dict, **kwargs):
    #     image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    #     num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    #     video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
    #     model_backbone = kwargs["model_backbone"]

    #     query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    #     for video_name, video_path, text in zip(batch_dict['id'], batch_dict["video_path"], batch_dict['sentence']):

    #         query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=text)])
    #         query_images.append([None])

    #         video_path = os.path.join(video_root, os.path.basename(video_path))
    #         frame_dir = os.path.join(frame_root, video_name)
    #         save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
    #         video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

    #         cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)])
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



# def load_youcook2_dataset(model_args, data_args, **kwargs):
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
