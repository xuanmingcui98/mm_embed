import os

import datasets
from ..dataset.base_pair_dataset import RESOLUTION_MAPPING, VideoDatasetProcessor, VideoSFTDatasetProcessor
from src.model.processor import VLM_VIDEO_TOKENS
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_EMBED_INSTRUCTION
from ..utils.vision_utils import process_video_frames
from ..loader.mixed_dataset import AutoPairDataset, AutoSFTDataset

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


VRET_QRY_PROMPT =  "Find a video that contains the following visual content: "
VRET_TGT_PROMPT =  "Understand the content of the provided video: "

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

        neg_text = [""]
        neg_image = [None]
        neg_description = [""]
        try:
            data_idx, data_id, conversations, video_id = idx, batch_dict['id'][idx], batch_dict['conversations'][idx], batch_dict['video'][idx]
            if data_mode == 'caption_retrieval':
                query, pos_text = process_conversations(conversations, video_token=VLM_VIDEO_TOKENS[model_backbone])

                if not self.data_args.fast_iter_with_no_visual:
                    frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
                    query_images = {"bytes": [None] * num_frames, "paths": frame_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames}
                
                if self.query_descriptions:
                    query_description = self.query_descriptions.get((video_id,))
                    # if not query_description:
                        # print(f'No query description found for ({video_id},) for dataset {self.dataset_config["dataset_name"]}')
                
                if self.data_args.hard_negative_dir:
                    hard_negative_samples = batch_dict['hard_negatives'][idx]
                    hard_negative_samples = [self.non_iter_dataset[i] for i in hard_negative_samples]
                    neg_text = [process_conversations(x['conversations'], video_token=VLM_VIDEO_TOKENS[model_backbone])[1] for x in hard_negative_samples]
                    neg_image = [None] * len(hard_negative_samples)
                    neg_description = [""] * len(hard_negative_samples)
            else:
                query = process_conversations_for_vret(conversations, prompt=VRET_QRY_PROMPT)
                if not self.data_args.fast_iter_with_no_visual:
                    frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
                    pos_images = {"bytes": [None] * num_frames, "paths": frame_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames}
                pos_text = VRET_TGT_PROMPT + VLM_VIDEO_TOKENS[model_backbone]
                if self.data_args.hard_negative_dir:
                    hard_negative_samples = batch_dict['hard_negatives'][idx]
                    hard_negative_samples = [self.non_iter_dataset[i] for i in hard_negative_samples]
                    neg_text = [VRET_TGT_PROMPT + VLM_VIDEO_TOKENS[model_backbone]] * len(hard_negative_samples)
                    neg_image = []
                    neg_description = [""] * len(hard_negative_samples)
                    if self.target_descriptions:
                        neg_description = []
                        for hn in hard_negative_samples:
                            frame_paths = process_video_frames(os.path.join(frame_basedir, hn['video']), num_frames=num_frames)
                            img = {"bytes": [None] * num_frames, "paths": frame_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames}
                            neg_image.append(img)
                            if self.target_descriptions:
                                neg_description.append(self.target_descriptions.get((hn['video'],)))

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
                "neg_text": neg_text, "neg_image": neg_image,
                "query_description": query_description,
                "target_description": target_description,
                "neg_description": neg_description}


@AutoSFTDataset.register(DATASET_PARSER_NAME)
class LLaVaHoundCaptionSFTDatasetProcessor(VideoSFTDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

    def _load_hf_dataset(self):
        return datasets.load_dataset("json", split="train", data_files=self.dataset_config['dataset_path'], streaming=False)

    def _process_one_sample(self, idx, batch_dict, *args, **kwargs):
        frame_basedir = kwargs['video_frame_basedir']
        data_mode = kwargs.get('data_mode', 'caption_retrieval')
        num_frames = kwargs['num_frames']


        data_idx, data_id, conversations, video_id = idx, batch_dict['id'][idx], batch_dict['conversations'][idx], batch_dict['video'][idx]
        if data_mode == 'caption_retrieval':
            query = conversations[0]["value"].replace("<video>", '').strip()
            images = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
            images = {"bytes": [None] * num_frames, "paths": images}
            if self.data_args.prompt_format == 'cot':
                answer = self.query_descriptions.get((video_id,))
            else:
                answer = conversations[1]["value"]

        else:
            images = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
            images = {"bytes": [None] * num_frames, "paths": images}
            answer = self.target_descriptions.get((video_id,))
            query = ''
        
        return {"prompt": query, "image": images, "answer": answer}
