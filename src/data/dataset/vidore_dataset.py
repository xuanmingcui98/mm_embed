from datasets import load_dataset
from PIL import Image
from datasets.features.image import image_to_bytes

from torch.jit import isinstance
from ..dataset.base_pair_dataset import RESOLUTION_MAPPING, VideoDatasetProcessor
from src.model.processor import VLM_IMAGE_TOKENS
from ..prompts import TEXT_EMBED_INSTRUCTION, IMAGE_QA_INSTRUCTION
from ..loader.mixed_dataset import AutoPairDataset


def process_query(query, prompt, image_token):
    if prompt:
        query = f'{prompt} {image_token} {query}'
    else:
        query = f'{image_token} {query}'
    return query

TASK_INST_TGT = "" #"Represent the following text:\n"

DATASET_PARSER_NAME = "vidore"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction("colpali_train_set",
    {'query': IMAGE_QA_INSTRUCTION,
     'target': TEXT_EMBED_INSTRUCTION})
class VidoReDatasetProcessor(VideoDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

    def _load_hf_dataset(self):
        kwargs = self.dataset_config
        dataset_name = kwargs.get("dataset_name", None)
        dataset_split = kwargs.get("dataset_split", "train")
        dataset_path = kwargs.get("dataset_path", None)

        if dataset_name:
            dataset = load_dataset("vidore/colpali_train_set", split=dataset_split)
        elif dataset_path:
            dataset = load_dataset("parquet", data_files=dataset_path, split="train")

        dataset = dataset.add_column("id", list(range(len(dataset))))
        return dataset

    def _process_one_sample(self, idx, batch_dict, *args, **kwargs):
        model_backbone = kwargs['model_backbone']
        image_resolution = kwargs['image_resolution']

        query, prompt, image, image_filename, answer, answer_type, source = \
            batch_dict['query'][idx], batch_dict['prompt'][idx], batch_dict['image'][idx], \
            batch_dict['image_filename'][idx], batch_dict['answer'][idx], batch_dict['answer_type'][idx], batch_dict['source'][idx]
        query = process_query(query, prompt=prompt, image_token=VLM_IMAGE_TOKENS[model_backbone])
        if isinstance(image, Image.Image):
            # BC, datasets==2.21.0
            image_bytes = image_to_bytes(image)
            path = image_filename
        elif type(image) is dict:
            # datasets==3.3.2
            image_bytes = image['bytes']
            path = image['path']
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        query_description = target_description = None
        if self.query_descriptions is not None:
            query_description = self.query_descriptions.get((batch_dict['id'][idx],))
            if query_description is None:
                print(f"No query description for id {batch_dict['id'][idx]} for {self.dataset_config['dataset_name']} dataset")
        if self.target_descriptions is not None:
            target_description = self.target_descriptions.get((batch_dict['id'][idx],))
            if target_description is None:
                print(f"No target description for id {batch_dict['id'][idx]} for {self.dataset_config['dataset_name']} dataset")

        return {
            "query_text": query,
            "query_image": {"bytes": [image_bytes], "paths": [path], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]},
            "pos_text": answer,
            "pos_image": None,
            "neg_text": "",
            "neg_image": None,
            "query_description": query_description,
            "target_description": target_description,
        }