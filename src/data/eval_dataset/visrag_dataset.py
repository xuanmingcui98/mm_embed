import os
import hashlib

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset, load_qrels_mapping
from src.model.processor import process_input_text
from ..loader.mixed_dataset import AutoPairDataset
from ..prompts import VISDOC_QA_RETRIEVAL_INSTRUCTION, VISDOC_EMBED_INSTRUCTION


# TASK_INST_QRY = "Find a document image that matches the given query:"
# TASK_INST_TGT = "Understand the content of the provided document image."
TASK_INST_QRY = ""
TASK_INST_TGT = ""


VISRAG_DATASETS = [
    "VisRAG_ArxivQA",
    "VisRAG_ChartQA",
    "VisRAG_MP-DocVQA",
    "VisRAG_SlideVQA",
]

DATASET_PARSER_NAME = "visrag"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction(VISRAG_DATASETS,
    {'query': VISDOC_QA_RETRIEVAL_INSTRUCTION,
     'target': VISDOC_EMBED_INSTRUCTION})
class VisRAGEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args,
                         query_key_text='query', query_key_mm=None, cand_key_text="query-id", cand_key_mm="query-id",
                            **dataset_config)

    def _load_hf_dataset(self):
        hf_dataset_name = EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']][0]
        hf_dataset_split = EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']][2]
        # BEIR format
        qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
        corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
        dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
        qrels_mapping = load_qrels_mapping(qrels)
        dataset = sample_dataset(dataset, **self.dataset_config)

        self.dataset_config['model_backbone'] = self.model_args.model_backbone
        self.dataset_config['image_resolution'] = self.data_args.image_resolution
        self.dataset_config['qrels_mapping'] = qrels_mapping
        corpus = corpus.map(lambda x: corpus_prepare(x, **self.dataset_config), batched=True,
                            batch_size=1024, num_proc=4,
                            drop_last_batch = False, load_from_cache_file=False)
        corpus = corpus.select_columns(['cand_text', 'cand_image', 'dataset_infos'])

        return dataset, corpus

    def _process_one_sample(self, data_idx, batch_dict, **kwargs):
        image_resolution = kwargs["image_resolution"]
        model_backbone   = kwargs["model_backbone"]
        qrels_mapping    = kwargs["qrels_mapping"]
        image_root       = kwargs["image_root"]

        # Fields
        query_id = batch_dict["query-id"][data_idx]
        query    = batch_dict["query"][data_idx]

        # Query: text-only
        query_text  = process_input_text(TASK_INST_QRY, model_backbone, text=query)
        query_image = None

        # Candidates (truncate filename + hash to create stored path)
        cand_text, cand_image = [], []
        cand_names, label_names, rel_scores = [], [], []

        for image_name, rel_score in qrels_mapping[query_id].items():
            base, ext = os.path.splitext(image_name)
            short_base = base[:50] + "_" + hashlib.md5(image_name.encode("utf-8")).hexdigest()[:8]
            new_imagename = short_base + ext
            image_path = f"{image_root}/{new_imagename}"
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path {image_path} not found.")

            cand_text.append(process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True))
            cand_image.append({
                "bytes": [None],
                "paths": [image_path],
                "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)],
            })
            cand_names.append(image_name)   # keep original (unshortened) name for metadata
            label_names.append(image_name)
            rel_scores.append(rel_score)

        dataset_info = {
            "cand_names": cand_names,
            "label_name": label_names,
            "rel_scores": rel_scores,
        }

        return {
            "query_text": query_text,     # str
            "query_image": query_image,   # None
            "cand_text": cand_text,       # list[str]
            "cand_image": cand_image,     # list[dict]
            "dataset_infos": dataset_info, 
        }

def corpus_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    cand_texts, cand_images, dataset_infos = [], [], []
    for image_name, image in zip(batch_dict['corpus-id'], batch_dict['image']):
        # some image_name are super long...
        base, ext = os.path.splitext(image_name)
        short_base = base[:50] + "_" + hashlib.md5(image_name.encode('utf-8')).hexdigest()[:8] # Truncate base, add original filename hash
        new_imagename = short_base + ext
        image_path = f'{image_root}/{new_imagename}'
        if not os.path.exists(image_path):
            os.makedirs(image_root, exist_ok=True)
            image.save(image_path)
        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True)])
        cand_images.append([ImageVideoInstance(
            bytes=[None],
            paths=[image_path],
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
        ).to_dict()])
        dataset_infos.append({
            "cand_names": [image_name],
        })

    return {"cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}