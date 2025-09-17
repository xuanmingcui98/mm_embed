import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset, load_qrels_mapping
from src.model.processor import process_input_text
from src.utils import print_master
from ..prompts import VISDOC_QA_RETRIEVAL_INSTRUCTION, VISDOC_EMBED_INSTRUCTION, VIDORE_QA_RETRIEVAL_DATASETS
from ..loader.mixed_dataset import AutoPairEvalDataset

TASK_INST_QRY = "Find a document image that matches the given query:"
TASK_INST_TGT = "Understand the content of the provided document image."
# ColPali models use no prompts
# TASK_INST_QRY = ""
# TASK_INST_TGT = ""



DATASET_PARSER_NAME = "vidore"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction(VIDORE_QA_RETRIEVAL_DATASETS,
    {'query': VISDOC_QA_RETRIEVAL_INSTRUCTION,
     'target': VISDOC_EMBED_INSTRUCTION})
class VidoreEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args,**dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args,
                         query_key_text="query", query_key_mm=None,
                         cand_key_text="query-id", cand_key_mm="video-id",
                         **dataset_config)
    
    def _load_hf_dataset(self):
        hf_dataset_name = EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']][0]
        hf_dataset_split = EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']][2]
        # BEIR format
        dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
        lang = EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']][1]
        if lang is not None:
            dataset = dataset.filter(lambda example: example["language"] == lang)
        qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
        corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
        qrels_mapping = load_qrels_mapping(qrels)
        dataset = sample_dataset(dataset, **self.dataset_config)


        if self.data_args.debug_prompt:
            dataset = dataset.select(range(1))
            corpus = corpus.select(range(1))
        print_master(f"Loaded {self.dataset_config['dataset_name']}")
        print_master(f"#hf_dataset_name={hf_dataset_name}")
        print_master(f"#hf_dataset_split={hf_dataset_split}")
        print_master(f"#queries={len(dataset)}")
        print_master(f"#cand={len(corpus)}")
        print_master(f"#qrels={len(qrels)}")

        self.dataset_config['model_backbone'] = self.model_args.model_backbone
        self.dataset_config['image_resolution'] = self.data_args.image_resolution
        self.dataset_config['qrels_mapping'] = qrels_mapping

        corpus = corpus.map(lambda x: self.corpus_prepare(x, **self.dataset_config), batched=True,
                            batch_size=128, #num_proc=8,
                            drop_last_batch=False, load_from_cache_file=False)
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

        # Candidates: from qrels per query_id
        cand_text, cand_image = [], []
        cand_names, label_names, rel_scores = [], [], []

        query_description = self.query_descriptions[query_id] if self.query_descriptions else None
        target_description = []

        for corpus_id, rel_score in qrels_mapping[query_id].items():
            image_path = f"{image_root}/{corpus_id}.png"
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path {image_path} not found.")
            cand_text.append(process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True))
            if self.target_descriptions:
                target_desc = self.target_descriptions.get((corpus_id,))
                if not target_desc:
                    print(f"No target description found for corpus_id {corpus_id} for {self.dataset_config['dataset_name']}")
                target_description.append(target_desc)
            else:
                target_description.append(None)
            cand_image.append(
                ImageVideoInstance(
                    bytes=[None],
                    paths=[image_path],
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
                ).to_dict()
            )
            cand_names.append(corpus_id)
            label_names.append(corpus_id)
            rel_scores.append(rel_score)

        dataset_info = {
            "cand_names": cand_names,
            "label_name": label_names,
            "rel_scores": rel_scores,
        }

        return {
            "query_text": query_text,    # str
            "query_image": query_image,  # None
            "cand_text": cand_text,      # list[str]
            "cand_image": cand_image,    # list[dict]
            "dataset_infos": dataset_info,
            "query_description": query_description,  # str or None
            "target_description": target_description,  # list[str] or None
        }

    def corpus_prepare(self, batch_dict, *args, **kwargs):
        image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
        image_root = kwargs['image_root']

        cand_texts, cand_images, dataset_infos = [], [], []
        for corpus_id, image in zip(batch_dict['corpus-id'], batch_dict['image']):
            image_path = f'{image_root}/{corpus_id}.png'
            if not os.path.exists(image_path):
                os.makedirs(image_root, exist_ok=True)
                image.save(image_path)

            if self.apply_chat_template:
                target_description = None
                if self.target_descriptions:
                    target_description = self.target_descriptions.get((corpus_id,))
                    if not target_description:
                        print(f"Warning: No target description found for corpus_id {corpus_id} for corpus dataset for {self.dataset_config['dataset_name']}")
                cand_texts.append([self.format_text_for_chat_template(
                    False, 
                    text=self.instruction['target'],
                    image_path=image_path,
                    add_generation_prompt=self.model_args.do_sft_target,
                    description=target_description
                    )])
            else:
                cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True)])
            cand_images.append([ImageVideoInstance(
                bytes=[None],
                paths=[image_path],
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
            ).to_dict()])
            dataset_infos.append({
                "cand_names": [corpus_id],
            })

        return {"cand_text": cand_texts, "cand_image": cand_images,
                "dataset_infos": dataset_infos}