import os

from datasets import Dataset
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset
from src.data.eval_dataset.video_classification_utils import VIDEOCLS_LABEL_MAPPING, DATASET_INSTRUCTION
from src.data.utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text
from ..prompts import TEXT_EMBED_INSTRUCTION
from ..loader.mixed_dataset import AutoPairEvalDataset


TASK_INST_TGT = "" # "Represent the following text:\n"

DATASET_INSTRUCTION = {
    'Kinetics-700': 'Recognize the category of the video content.',
    'UCF101': 'What activities or sports are being performed by the person in the video?',
    'HMDB51': 'What actions or objects interactions are the person in the video doing?',
    'Breakfast': 'Recognize the breakfast type that the person is cooking in the video. ',
}

DATASET_PARSER_NAME = "video_classification"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction("HMDB51",
    {'query': """Given the video, identify the actions or object interactions being performed by the person. Embed the video with your answer.""",
     'target': TEXT_EMBED_INSTRUCTION})
@AutoPairEvalDataset.register_instruction("UCF101",
    {'query': """Given the video, identify the activities or sports being performed by the person. Embed the video with your answer.""",
     'target': TEXT_EMBED_INSTRUCTION})
@AutoPairEvalDataset.register_instruction("Kinetics-700",
    {'query': """Given the video, identify the category of the video content. Embed the video with your answer.""",
     'target': TEXT_EMBED_INSTRUCTION})
@AutoPairEvalDataset.register_instruction("Breakfast",
    {'query': """Given the video, identify the breakfast type that the person is cooking in the video. Embed the video with your answer.""",
     'target': TEXT_EMBED_INSTRUCTION})
class VideoClassificationEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__("video_classification", *args,
                         query_key_text=None, query_key_mm='video_id',
                         cand_key_text='pos_text', cand_key_mm=None,
                         **dataset_config)

    def _load_hf_dataset(self):
        dataset_name = self.dataset_config['dataset_name']
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        corpus = Dataset.from_list([{
            "cand_text": [self.format_text_for_chat_template(False, text=label)],
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

        query_description = None
        if self.query_descriptions:
            query_description = self.query_descriptions.get((video_id,))
            if query_description is None:
                print(f"No query description for video {video_id} in {self.dataset_config['dataset_name']} dataset")

        return {
            "query_text": query_text,   # str
            "query_image": query_image, # dict or None
            "cand_text": cand_text,     # [label]
            "cand_image": cand_image,   # [None]
            "dataset_infos": dataset_infos,
            "query_description": query_description,
            "target_description": None,
        }