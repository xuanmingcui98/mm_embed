import os

import datasets
from ..dataset.base_pair_dataset import RESOLUTION_MAPPING, VideoDatasetProcessor
from src.model.processor import VLM_VIDEO_TOKENS
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_EMBED_INSTRUCTION
from ..utils.vision_utils import process_video_frames
from ..loader.mixed_dataset import AutoPairDataset

def process_conversations(conversations, video_token):
    query = conversations[0]["value"].replace("<video>", ''.join([video_token]))
    pos_text = conversations[1]["value"]
    return query, pos_text

def process_conversations_for_vret(conversations, prompt):
    caption = conversations[1]["value"]
    query = caption
    if prompt:
        query = prompt + query

    return query


VRET_QRY_PROMPT = "" # "Find a video that contains the following visual content: "
VRET_TGT_PROMPT = "" # "Understand the content of the provided video: "

DATASET_PARSER_NAME = "llavahound_caption"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction("video_caption_300k-t2v",
    {'query': TEXT_EMBED_INSTRUCTION,
    'target': VIDEO_EMBED_INSTRUCTION})
@AutoPairDataset.register_instruction("video_caption_300k-v2t",
    {'query': VIDEO_EMBED_INSTRUCTION,
    'target': TEXT_EMBED_INSTRUCTION})
class LLaVaHoundCaptionDatasetProcessor(VideoDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

    def _load_hf_dataset(self):
        return datasets.load_dataset("json", split="train", data_files=self.dataset_config['dataset_path'], streaming=False)

    def _process_one_sample(self, idx, batch_dict, *args, **kwargs):
        model_backbone = kwargs['model_backbone']
        image_resolution = kwargs['image_resolution']
        frame_basedir = kwargs['video_frame_basedir']
        data_mode = kwargs.get('data_mode', 'caption_retrieval')
        num_frames = kwargs['num_frames']

        query_images = pos_images = None
        query_description = target_description = None
        try:
            data_idx, data_id, conversations, video_id = idx, batch_dict['id'][idx], batch_dict['conversations'][idx], batch_dict['video'][idx]
            if data_mode == 'caption_retrieval':
                query, pos_text = process_conversations(conversations, video_token=VLM_VIDEO_TOKENS[model_backbone])
                frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
                query_images = {"bytes": [None] * num_frames, "paths": frame_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames}
                
                if self.query_descriptions:
                    query_description = self.query_descriptions.get((video_id,))
                    if not query_description:
                        print(f'No query description found for ({video_id},) for dataset {self.dataset_config["dataset_name"]}')

            else:
                query = process_conversations_for_vret(conversations, prompt=VRET_QRY_PROMPT)
                frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
                pos_images = {"bytes": [None] * num_frames, "paths": frame_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames}
                pos_text = VRET_TGT_PROMPT + VLM_VIDEO_TOKENS[model_backbone]
                if self.target_descriptions:
                    target_description = self.target_descriptions.get((video_id,))
                    if not target_description:
                        print(f'No target description found for ({video_id},) for dataset {self.dataset_config["dataset_name"]}')


        except Exception as e:
            print(f'Error in processing {DATASET_PARSER_NAME}: \n\t\tdata id: {data_id} \n\t\tconversations: {conversations}')
            print(e)
            raise e
    
        return {"query_text": query, "query_image": query_images,
                "pos_text": pos_text, "pos_image": pos_images,
                "neg_text": "", "neg_image": None,
                "query_description": query_description,
                "target_description": target_description}