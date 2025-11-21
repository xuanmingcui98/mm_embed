import os

import datasets
from ..dataset.base_pair_dataset import RESOLUTION_MAPPING, VideoDatasetProcessor, VideoSFTDatasetProcessor
from src.model.processor import VLM_VIDEO_TOKENS
from ..utils.vision_utils import process_video_frames
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_EMBED_INSTRUCTION, VIDEO_QA_INSTRUCTION
from ..loader.mixed_dataset import AutoPairDataset, AutoSFTDataset

def process_conversations(conversations, video_token, prompt):
    query = conversations[0]["value"].replace("<video>", ''.join([video_token]))
    if prompt:
        query = prompt + query
    pos_text = conversations[1]["value"]
    return query, pos_text

QA_QUERY_PROMPT =  "Answer a question based on the content of a video. "
TASK_INST_TGT = "Represent the following text:\n"

DATASET_PARSER_NAME = "llavahound_qa"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction("video_qa_240k",
    {'query': VIDEO_QA_INSTRUCTION,
    'target': TEXT_EMBED_INSTRUCTION})
class LLaVaHoundQADatasetProcessor(VideoDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

    def _load_hf_dataset(self):
        return datasets.load_dataset("json", split="train", data_files=self.dataset_config['dataset_path'], streaming=False)

    def _process_one_sample(self, idx, batch_dict, *args, **kwargs):
        model_backbone = kwargs['model_backbone']
        image_resolution = kwargs['image_resolution']
        frame_basedir = kwargs['video_frame_basedir']
        num_frames = kwargs['num_frames']

        neg_text = [""]
        neg_image = [None]
        neg_description = [""]

        try:
            data_idx, data_id, conversations, video_id = idx, batch_dict['id'][idx], batch_dict['conversations'][idx], batch_dict['video'][idx]
            query, pos_text = process_conversations(conversations, video_token=VLM_VIDEO_TOKENS[model_backbone], prompt=QA_QUERY_PROMPT)
            if not self.data_args.fast_iter_with_no_visual:
                frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
                video_frames = {"bytes": [None] * num_frames, "paths": frame_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames}
            else:
                video_frames = None

            query_description = None
            if self.query_descriptions is not None:
                query_description = self.query_descriptions.get((data_id,))
                # if not query_description:
                    # print(f"No target description for video {data_id} in {self.dataset_config['dataset_name']} dataset")

            if self.data_args.hard_negative_dir:
                hard_negative_samples = batch_dict['hard_negatives'][idx]
                hard_negative_samples = [self.non_iter_dataset[i] for i in hard_negative_samples]
                neg_text = [process_conversations(x['conversations'], video_token=VLM_VIDEO_TOKENS[model_backbone], prompt=QA_QUERY_PROMPT)[1] for x in hard_negative_samples]
                neg_image = [None] * len(hard_negative_samples)
                neg_description= [""] * len(hard_negative_samples)
                

            return {"query_text": query, "query_image": video_frames,
                    "pos_text": pos_text, "pos_image": None,
                    "neg_text": neg_text, "neg_image": neg_image,
                    "neg_description": neg_description,
                    "query_description": query_description,
                    "target_description": None}
        except Exception as e:
            print(f'Error in processing {DATASET_PARSER_NAME}: \n\t\tdata id: {data_id} \n\t\tconversations: {conversations}')
            print(e)
            raise e


@AutoSFTDataset.register(DATASET_PARSER_NAME)
class LLaVaHoundQASFTDatasetProcessor(VideoSFTDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

    def _load_hf_dataset(self):
        return datasets.load_dataset("json", split="train", data_files=self.dataset_config['dataset_path'], streaming=False)
    
    def _process_one_sample(self, idx, batch_dict, *args, **kwargs):
        model_backbone = kwargs['model_backbone']
        image_resolution = kwargs['image_resolution']
        frame_basedir = kwargs['video_frame_basedir']
        num_frames = kwargs['num_frames']

        try:
            data_idx, data_id, conversations, video_id = idx, batch_dict['id'][idx], batch_dict['conversations'][idx], batch_dict['video'][idx]
            query, pos_text = conversations[0]["value"].replace("<video>", '').strip(), conversations[1]["value"].strip()
            images = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
            answer = self.query_descriptions.get((data_id,))

            return {"prompt": query, 
                    "image": {"bytes": [None] * num_frames, "paths": images}, 
                    "answer": answer}
        
        except Exception as e:
            print(f'Error in processing {DATASET_PARSER_NAME}: \n\t\tdata id: {data_id} \n\t\tconversations: {conversations}')
            print(e)
            raise e
