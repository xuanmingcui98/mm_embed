import os
import os.path as osp
import json
from datasets import load_from_disk, Dataset
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.model.processor import process_input_text, VLM_IMAGE_TOKENS
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_EMBED_INSTRUCTION
from ..loader.mixed_dataset import AutoPairEvalDataset

TASK_INST_QRY = """Given a COCO-style caption, first use your knowledge to expand it into a more detailed and concise description of the target image, then generate a summarization based on the description. Let's think step by step.

Caption: {query}"""
TASK_INST_TGT = "Given an image, first generate a detailed and informative description of the image, then generate a COCO-style caption based on the description. Let's think step by step."
# TASK_INST_QRY = ""
# TASK_INST_TGT = ""


def create_xtd_dataset(path):
    dataset = []
    xtd_path = osp.join(path, "XTD10")
    images = open(osp.join(xtd_path, "test_image_names.txt")).read().splitlines()
    for lang_file in os.listdir(xtd_path):
        if lang_file.startswith("test_1k"):
            lang = lang_file.split(".")[0].split("_")[-1]
            texts = open(osp.join(xtd_path, lang_file)).read().splitlines()
            for text, image in zip(texts, images):
                dataset.append({"query_text": text, "query_image": image, "lang": lang})

    mic_path = osp.join(path, "MIC")
    for lang_file in os.listdir(mic_path):
        if lang_file.startswith("test_1k"):
            lang = lang_file.split(".")[0].split("_")[-1]
            texts = open(osp.join(mic_path, lang_file)).read().splitlines()
            for text, image in zip(texts, images):
                dataset.append({"query_text": text, "query_image": image, "lang": lang})
    
    texts_jp = open(osp.join(path, "STAIR", "test_1kcaptions_jp.txt")).read().splitlines()
    for text, image in zip(texts_jp, images):
        dataset.append({"query_text": text, "query_image": image, "lang": "jp"})
    
    dataset = Dataset.from_list(dataset)
    return dataset
    

def create_crossmodal3600_dataset(anno):
    langs = ['ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fil', 'fr', 'hi', 'hr', 'hu', 'id', 'it', 'he', 'ja', 'ko', 'mi', 'nl', 'no', 'pl', 'pt', 'quz', 'ro', 'ru', 'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh']
    anno = [json.loads(x) for x in open(anno)]
    dataset = []
    for row in anno:
        for lang in langs:
            if not row[lang]['caption'][0]:
                continue

            dataset.append({
                "query_text": row[lang]['caption'][0],
                "query_image": row['image/key'],
                "lang": lang
            })
    dataset = Dataset.from_list(dataset)
    return dataset


DATASET_PARSER_NAME = "multilingual_t2i"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
class MultiLingualT2IEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

    def _load_hf_dataset(self):
        if self.dataset_config.get("dataset_path"):
            return load_from_disk(self.dataset_config.get("dataset_path")), None
        return load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']]), None

    def _process_one_sample(self, data_idx, batch_dict, **kwargs):
        image_resolution = kwargs["image_resolution"]
        model_backbone   = kwargs["model_backbone"]
        image_root       = kwargs["image_root"]

        # Fields for this sample
        image_name = batch_dict["target_image"][data_idx]
        caption = batch_dict["query_text"][data_idx]

        # Query: text only (no query-side image/video)
        query_text  = TASK_INST_QRY.format(query=caption)
        query_image = None

        target_description = None
        if self.target_descriptions:
            target_description = self.target_descriptions.get((image_name,))
            if not target_description:
                print(f'No target description found for ({image_name},) for dataset {self.dataset_config["dataset_name"]}')


        if ".jpg" not in image_name:
            image_name += ".jpg"
        cand_text = VLM_IMAGE_TOKENS[model_backbone] + "\n" + TASK_INST_TGT
        cand_image = [{
            "bytes": [None],
            "paths": [os.path.join(image_root, image_name)],
            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)],
        }]

        dataset_info = {
            "query_id": (caption,),
            "cand_names": [image_name],
            "label_name": image_name,
        }

        return {
            "query_text": query_text,
            "query_image": query_image,
            "cand_text": cand_text,
            "cand_image": cand_image,
            "dataset_infos": dataset_info,
            "query_description": None,
            "target_description": target_description,
        }
