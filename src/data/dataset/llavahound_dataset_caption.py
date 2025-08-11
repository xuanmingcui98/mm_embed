import os

import datasets
from ..dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING, VideoDatasetProcessor
from src.model.processor import VLM_VIDEO_TOKENS
from ..utils.vision_utils import process_video_frames


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


VRET_QRY_PROMPT = "Find a video that contains the following visual content: "
VRET_TGT_PROMPT = "Understand the content of the provided video: "

TASK_INST_TGT = "Represent the following text:\n"

DATASET_PARSER_NAME = "llavahound_caption"
@AutoPairDataset.register(DATASET_PARSER_NAME)
class LLaVaHoundCaptionDatasetProcessor(VideoDatasetProcessor):
    def __init__(self, model_args, data_args, training_args, **kwargs):
        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, 
                         query_key_text="conversations",
                         query_key_mm="video",
                         cand_key_text="conversations",
                         cand_key_mm="video",
                         **kwargs)

    def _load_hf_dataset(self):
        return datasets.load_dataset("json", split="train", data_files=self.dataset_config['dataset_path'], streaming=False)

    # def _add_signature_columns_map_func(self, batch_dict):
    #     signature_columns = {
    #         "query_key_text": batch_dict['conversations'],
    #         "query_key_mm": batch_dict['video'],
    #         "cand_key_text": batch_dict['conversations'],
    #         "cand_key_mm": batch_dict['video_id']}
    #     return batch_dict | signature_columns

    def _process_one_sample(self, idx, batch_dict, *args, **kwargs):
        model_backbone = kwargs['model_backbone']
        image_resolution = kwargs['image_resolution']
        frame_basedir = kwargs['video_frame_basedir']
        data_mode = kwargs.get('data_mode', 'caption_retrieval')
        num_frames = kwargs['num_frames']

        query_images = pos_images = None

        try:
            data_idx, data_id, conversations, video_id = idx, batch_dict['id'][idx], batch_dict['conversations'][idx], batch_dict['video'][idx]
            if data_mode == 'caption_retrieval':
                query, pos_text = process_conversations(conversations, video_token=VLM_VIDEO_TOKENS[model_backbone])
                frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
                query_images = {"bytes": [None] * num_frames, "paths": frame_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames}


            else:
                query = process_conversations_for_vret(conversations, prompt=VRET_QRY_PROMPT)
                frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
                pos_images = {"bytes": [None] * num_frames, "paths": frame_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames}
                pos_text = VRET_TGT_PROMPT + VLM_VIDEO_TOKENS[model_backbone]
        except Exception as e:
            print(f'Error in processing {DATASET_PARSER_NAME}: \n\t\tdata id: {data_id} \n\t\tconversations: {conversations}')
            print(e)
            raise e
    
        return {"query_text": query, "query_image": query_images,
                "pos_text": pos_text, "pos_image": pos_images,
                "neg_text": "", "neg_image": None}

        # return {"query_text": query_texts, "query_image": query_images,
        #         "pos_text": pos_texts, "pos_image": pos_images,
        #         "neg_text": neg_texts, "neg_image": neg_images}

    # @add_metainfo_hook
    # def batch_preprocess(self, batch_dict, *args, **kwargs):
    #     model_backbone = kwargs['model_backbone']
    #     image_resolution = kwargs['image_resolution']
    #     frame_basedir = kwargs['video_frame_basedir']
    #     data_mode = kwargs.get('data_mode', 'caption_retrieval')
    #     num_frames = kwargs['num_frames']
    #     batch_size = len(batch_dict['id'])
    #     query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    #     for data_idx, (data_id, conversations, video_id) in enumerate(zip(batch_dict['id'], batch_dict['conversations'], batch_dict['video'])):
    #         try:
    #             if data_mode == 'caption_retrieval':
    #                 query, pos_text = process_conversations(conversations, video_token=VLM_VIDEO_TOKENS[model_backbone])
    #                 frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
    #                 video_frames = {"bytes": [None] * num_frames, "paths": frame_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames}

    #                 if self.data_args.apply_chat_template:
    #                     query = self.format_text_for_chat_template(is_query=True, text=query, video_path=video_id, key=(query, video_id))
    #                     pos_text = self.format_text_for_chat_template(is_query=False, text=pos_text, key=(pos_text, ""))

    #                 query_texts.append(query)
    #                 pos_texts.append(pos_text)
    #                 neg_texts.append("")
    #                 query_images.append(video_frames)
    #                 pos_images.append(None)
    #                 neg_images.append(None)
    #             elif data_mode == 'video_retrieval':
    #                 query = process_conversations_for_vret(conversations, prompt=VRET_QRY_PROMPT)
    #                 frame_paths = process_video_frames(os.path.join(frame_basedir, video_id), num_frames=num_frames)
    #                 video_frames = {"bytes": [None] * num_frames, "paths": frame_paths, "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames}

    #                 if self.data_args.apply_chat_template:
    #                     query = self.format_text_for_chat_template(is_query=True, text=query, key=(query, ""))
    #                     pos_text = self.format_text_for_chat_template(is_query=False, text=VRET_TGT_PROMPT, video_path=video_id, key=(VRET_TGT_PROMPT, video_id))
    #                 else:
    #                     pos_text = VRET_TGT_PROMPT + VLM_VIDEO_TOKENS[model_backbone]
    #                 query_texts.append(query)
    #                 pos_texts.append(pos_text)
    #                 neg_texts.append("")
    #                 query_images.append(None)
    #                 pos_images.append(video_frames)
    #                 neg_images.append(None)
    #             else:
    #                 raise NotImplementedError(f'data_mode={data_mode} not implemented.')
    #         except Exception as e:
    #             print(f'Error in processing {DATASET_PARSER_NAME}: \n\t\tdata id: {data_id} \n\t\tconversations: {conversations}')
    #             print(e)
    #             raise e
    #     # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    #     return {"query_text": query_texts, "query_image": query_images,
    #             "pos_text": pos_texts, "pos_image": pos_images,
    #             "neg_text": neg_texts, "neg_image": neg_images}



# def load_llavahound_caption_dataset(model_args, data_args, training_args, *args, **kwargs):
#     dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
#     assert "dataset_path" in kwargs, "`dataset_path` should be given for loading llavahound dataset."
#     assert "num_frames" in kwargs, "`num_frames` should be given for loading llavahound dataset."
#     assert "video_frame_basedir" in kwargs, "`video_frame_basedir` should be given for loading llavahound dataset."
#     dataset_path = kwargs["dataset_path"]
#     dataset = datasets.load_dataset("json", split="train", data_files=dataset_path, streaming=False)
#     num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
#     if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
#         num_rows = int(num_sample_per_subset)
#         dataset = dataset.select(range(num_rows))
#     num_rows = dataset.num_rows

#     num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
#     dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards

#     kwargs['model_backbone'] = model_args.model_backbone
#     kwargs['image_resolution'] = data_args.image_resolution
#     kwargs['video_frame_basedir'] = kwargs["video_frame_basedir"]
#     kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{dataset_name}'
#     # dataset = dataset.shuffle(buffer_size=8192, seed=training_args.seed)
#     dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=128, drop_last_batch=True)
#     dataset = dataset.cast(MULTIMODAL_FEATURES)

#     # num_rows in iterable_dataset is not available, set it here for printing dataset stats
#     setattr(dataset, 'num_rows', num_rows)
#     return dataset
