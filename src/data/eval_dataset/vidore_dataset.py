import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset, load_qrels_mapping
from src.model.processor import process_input_text
from src.utils import print_master

# TASK_INST_QRY = "Find a document image that matches the given query:"
# TASK_INST_TGT = "Understand the content of the provided document image."
# ColPali models use no prompts
TASK_INST_QRY = ""
TASK_INST_TGT = ""

DATASET_PARSER_NAME = "vidore"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
class VidoreEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self,                 
                 model_args, 
                 data_args, 
                 training_args, 
                 processor, 
                 **dataset_config):

        super().__init__(DATASET_PARSER_NAME, model_args, data_args, training_args, processor, 
                         query_key_text="query", query_key_mm=None,
                         cand_key_text="query-id", cand_key_mm="video-id",
                         **dataset_config)

    def _add_signature_columns_map_func(self, batch_dict):
        signature_columns = {

            # @xuanming we assume modality in the order of text, image, video
            # current assume two modalities max for query and target
            "query_key_text": batch_dict['question'],
            "query_key_mm": [os.path.join(subset, video_id) for subset, video_id in zip(batch_dict['subset'], batch_dict['id'])],
            "cand_key_text": [""] * len(batch_dict['question']),
            "cand_key_mm": [""] * len(batch_dict['question'])}
        return batch_dict | signature_columns
    
    def _load_hf_dataset(self):
        hf_dataset_name = EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']][0]
        hf_dataset_split = EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']][2]
        lang = EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']][1]
        # BEIR format
        dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
        if lang is not None:
            dataset = dataset.filter(lambda example: example["language"] == lang)
        qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
        corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
        qrels_mapping = load_qrels_mapping(qrels)
        dataset = sample_dataset(dataset, **self.dataset_config)
        print_master(f"Loaded {self.dataset_config['dataset_name']}")
        print_master(f"#hf_dataset_name={hf_dataset_name}")
        print_master(f"#hf_dataset_split={hf_dataset_split}")
        print_master(f"#queries={len(dataset)}")
        print_master(f"#cand={len(corpus)}")
        print_master(f"#qrels={len(qrels)}")

        self.dataset_config['model_backbone'] = self.model_args.model_backbone
        self.dataset_config['image_resolution'] = self.data_args.image_resolution
        self.dataset_config['qrels_mapping'] = qrels_mapping

        corpus = corpus.map(lambda x: corpus_prepare(x, **self.dataset_config), batched=True,
                            batch_size=2048, #num_proc=8,
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

        for corpus_id, rel_score in qrels_mapping[query_id].items():
            image_path = f"{image_root}/{corpus_id}.png"
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path {image_path} not found.")
            cand_text.append(process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True))
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
        }


    # @add_metainfo_hook
    # def batch_preprocess(self, batch_dict, **kwargs):
    #     image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    #     qrels_mapping = kwargs['qrels_mapping']
    #     image_root = kwargs['image_root']

    #     query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    #     for query_id, query in zip(batch_dict['query-id'], batch_dict['query']):
    #         query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=query)])
    #         query_images.append([None])
    #         cand_text, cand_image, cand_names, label_names = [], [], [], []
    #         rel_scores = []

    #         for corpus_id, rel_score in qrels_mapping[query_id].items():
    #             image_path = f'{image_root}/{corpus_id}.png'
    #             if not os.path.exists(image_path):
    #                 raise FileNotFoundError(f'Image path {image_path} not found.')
    #             cand_text.append(process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True))
    #             cand_image.append(ImageVideoInstance(
    #                 bytes=[None],
    #                 paths=[image_path],
    #                 resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
    #             ).to_dict())
    #             cand_names.append(corpus_id)
    #             label_names.append(corpus_id)
    #             rel_scores.append(rel_score)
    #         cand_texts.append(cand_text)
    #         cand_images.append(cand_image)
    #         dataset_infos.append({
    #                 "cand_names": cand_names,
    #                 "label_name": label_names,
    #                 "rel_scores": rel_scores,
    #         })

    #     processed_batch = {"query_text": query_texts, "query_image": query_images,
    #             "cand_text": cand_texts, "cand_image": cand_images,
    #             "dataset_infos": dataset_infos}
    #     return batch_dict | processed_batch


def corpus_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    cand_texts, cand_images, dataset_infos = [], [], []
    for corpus_id, image in zip(batch_dict['corpus-id'], batch_dict['image']):
        image_path = f'{image_root}/{corpus_id}.png'
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
            "cand_names": [corpus_id],
        })

    return {"cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


# def load_vidore_dataset(model_args, data_args, **kwargs):
#     hf_dataset_name = EVAL_DATASET_HF_PATH[kwargs['dataset_name']][0]
#     hf_dataset_split = EVAL_DATASET_HF_PATH[kwargs['dataset_name']][2]
#     lang = EVAL_DATASET_HF_PATH[kwargs['dataset_name']][1]
#     # BEIR format
#     dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
#     if lang is not None:
#         dataset = dataset.filter(lambda example: example["language"] == lang)
#     qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
#     corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
#     qrels_mapping = load_qrels_mapping(qrels)
#     dataset = sample_dataset(dataset, **kwargs)
#     print_master(f"Loaded {kwargs['dataset_name']}")
#     print_master(f"#hf_dataset_name={hf_dataset_name}")
#     print_master(f"#hf_dataset_split={hf_dataset_split}")
#     print_master(f"#queries={len(dataset)}")
#     print_master(f"#cand={len(corpus)}")
#     print_master(f"#qrels={len(qrels)}")

#     kwargs['model_backbone'] = model_args.model_backbone
#     kwargs['image_resolution'] = data_args.image_resolution
#     kwargs['qrels_mapping'] = qrels_mapping

#     corpus = corpus.map(lambda x: corpus_prepare(x, **kwargs), batched=True,
#                         batch_size=2048, num_proc=8,
#                         drop_last_batch=False, load_from_cache_file=False)
#     corpus = corpus.select_columns(['cand_text', 'cand_image', 'dataset_infos'])
#     dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
#                           batch_size=2048, num_proc=8,
#                           drop_last_batch=False, load_from_cache_file=False)
#     dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

#     return dataset, corpus
