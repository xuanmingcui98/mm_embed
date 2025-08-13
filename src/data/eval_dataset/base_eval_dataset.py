from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from abc import ABCMeta, abstractmethod
from functools import wraps
import pickle, os
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, Sequence
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.utils import print_master, print_rank
# from torch.utils.data import Dataset
from ..prompts import (get_query, get_target, 
                       IMAGE_TASKS, VIDEO_TASKS, VISDOC_TASKS,
                       format_description, format_text_for_chat_template, 
                       extract_query, extract_target)
import torch
from ...model.processor import process_input_text
from ..utils.dataset_utils import load_hf_dataset, sample_dataset
from ..dataset_hf_path import EVAL_DATASET_HF_PATH
from ...model.processor import VLM_VIDEO_TOKENS
from .video_classification_utils import VIDEOCLS_LABEL_MAPPING, DATASET_INSTRUCTION
from src.data.utils.vision_utils import save_frames, process_video_frames


# Schema for evaluation dataset, not used in the code.
EVAL_QRY_FEATURES = Features(**{
    "query_text": Sequence(Value(dtype='string')),  # Only one element, but make it as a sequence for collator usage
    "query_image": Sequence({
        "paths": Sequence(Value(dtype='string')),  # List of image paths (frames)
        "bytes": Sequence(Value(dtype='binary')),  # List of pre-saved image bytes
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))  # List of [width, height] pairs
    }),
    "cand_text": Sequence(Value(dtype='string')),  # Here it's only for hard negatives.
    "cand_video": Sequence({
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    }),
    "dataset_infos": {
        "cand_names": Sequence(Value(dtype='string')),
        "label_name": Value(dtype='string')
    }
})

EVAL_CAND_FEATURES = Features(**{
    "cand_text": Sequence(Value(dtype='string')),  # Only one element, but make it as a sequence for collator usage
    "cand_image": Sequence({
        "paths": Sequence(Value(dtype='string')),  # List of image paths (frames)
        "bytes": Sequence(Value(dtype='binary')),  # List of pre-saved image bytes
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))  # List of [width, height] pairs
    }),
    "dataset_infos": {
        "cand_name": Value(dtype='string'),
    },
})

RESOLUTION_MAPPING = {
    "high": (1344, 1344),
    "mid": (672, 672),
    "low": (128, 128),
}


class ImageVideoInstance:
    """
    len(bytes) == len(path) == len(resolution) == 1: image
    len(bytes) == len(path) == len(resolution) > 1: multi-image / video
    """
    def __init__(self, bytes, paths, resolutions):
        assert len(bytes) == len(paths) == len(resolutions)
        self.bytes = bytes
        self.paths = paths
        self.resolutions = resolutions

    def to_dict(self):
        return {
            "bytes": self.bytes,
            "paths": self.paths,
            "resolutions": self.resolutions,
        }


class AutoEvalPairDataset(metaclass=ABCMeta):
    # Base class for auto datasets.
    registry = {}

    def __init_subclass__(cls):
        if cls.__name__ not in AutoEvalPairDataset.registry:
            AutoEvalPairDataset.registry[cls.__name__] = cls
        else:
            raise RuntimeError('Subclass "{cls.__name__}" has already defined.')

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def instantiate(cls, dataset_parser, *args, **kwargs):
        try:
            return cls.registry[dataset_parser](*args, **kwargs).load()
        except Exception as e:
            raise e

    @classmethod
    def register(cls, dataset_name):
        def inner_wrapper(wrapped_class):
            if dataset_name in cls.registry:
                print(f"[Alert] AutoPairDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                # print(f"Adding {dataset_name}")
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @abstractmethod
    def main(self):
        pass


def add_metainfo_hook(f):
    """
    A post-processing wrapper function that add meta information (e.g. data_type, dataset_name, loss_type) into batches
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # go through data pipeline customized to each dataset
        batch_data = f(*args, **kwargs)
        # append common metadata
        batch_size = len(batch_data.get('query_text', batch_data.get('cand_text', [])))
        global_dataset_name = kwargs.get("global_dataset_name", "None")
        batch_data['global_dataset_name'] = [global_dataset_name] * batch_size
        return batch_data

    return wrapper


def generate_cand_dataset(dataset, corpus):
    """
    Used for generating candidate datasets.
    Flatten candidates, merge with corpus, deduplication
    """
    cand_rows = []
    all_cand_name = set()
    for row in dataset:
        assert len(row["cand_text"]) == len(row["cand_image"]) == len(row["dataset_infos"]["cand_names"])
        for cand_text, cand_image, cand_name in zip(row["cand_text"], row["cand_image"], row["dataset_infos"]["cand_names"]):
            if cand_name not in all_cand_name:
                cand_rows.append({
                    "cand_text": [cand_text],
                    "cand_image": [cand_image],
                    "dataset_infos": {"cand_name": cand_name},
                })
                all_cand_name.add(cand_name)

    if corpus is not None:
        for row in corpus:
            assert len(row["cand_text"]) == len(row["cand_image"]) == len(row["dataset_infos"]["cand_names"]) == 1
            cand_name = row["dataset_infos"]["cand_names"][0]
            if cand_name not in all_cand_name:
                cand_rows.append({
                    "cand_text": row["cand_text"],
                    "cand_image": row["cand_image"],
                    "dataset_infos": {"cand_name": row["dataset_infos"]["cand_names"][0]},
                })
                all_cand_name.add(cand_name)

    cand_dataset = Dataset.from_list(cand_rows)
    return cand_dataset

class BaseEvalDatasetProcessor:

    def __init__(self,
                 data_parser_name,
                 model_args, 
                 data_args, 
                 training_args, 
                 processor,
                 query_key_text=None, query_key_mm=None,
                 cand_key_text=None, cand_key_mm=None,
                 **dataset_config):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.processor = processor
        self.dataset_config = dataset_config
        self.data_parser_name = data_parser_name
        self.query_key_text = query_key_text
        self.query_key_mm = query_key_mm
        self.cand_key_text = cand_key_text
        self.cand_key_mm = cand_key_mm

        self.dataset_name = self.dataset_config.get("dataset_name")
        self.dataset_split = self.dataset_config.get("dataset_split", "test")
        self.dataset_config['model_backbone'] = model_args.model_backbone
        self.dataset_config['image_resolution'] = data_args.image_resolution
        self.apply_chat_template = data_args.apply_chat_template

        self.model_backbone = self.dataset_config['model_backbone']

        if self.model_args.meta_queries is not None and self.model_args.meta_queries > 0:
            self.meta_queries = "".join(
                [f'<meta_query_{i}>' for i in range(self.model_args.meta_queries)]
            )
        else:
            self.meta_queries = ''

        self.query_descriptions = self.target_descriptions = None
        if data_args.query_description_dir is not None and not model_args.do_sft_query:
            desc_path = os.path.join(data_args.query_description_dir, self.subset_name, "cot", "query.pkl")
            if os.path.exists(desc_path):
                with open(desc_path, "rb") as f:
                    self.query_descriptions = pickle.load(f)


        if data_args.target_description_dir is not None and not model_args.do_sft_target:
            desc_path = os.path.join(data_args.target_description_dir, self.subset_name, "cot", "target.pkl")
            if os.path.exists(desc_path):
                with open(desc_path, "rb") as f:
                    self.target_descriptions = pickle.load(f)

        self.target_cache = {}

    # def _add_signature_columns_map_func(self, batch_dict):
        
    #     raise NotImplementedError(
    #         f"{self.__class__.__name__} does not implement `_add_signature_columns_map_func` method. "
    #         "Please implement it in the subclass."
    #     )

    # def add_signature_columns(self):

    #     self.dataset = self.dataset.map(
    #         self._add_signature_columns_map_func,
    #         batched=True,
    #         batch_size=2048,
    #         num_proc=4,
    #         drop_last_batch=False,
    #         load_from_cache_file=False
    #     )


    def _load_hf_dataset(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `_load_hf_dataset` method. "
            "Please implement it in the subclass."
        )

    def load(self):

        self.dataset, self.corpus = self._load_hf_dataset()
        # debug: take 1 row
        if self.data_args.debug_prompt:
            self.dataset = self.dataset.select(range(1))
        self.dataset = sample_dataset(self.dataset, **self.dataset_config)
        # self.add_signature_columns()
        # if self.data_args.apply_chat_template:
        #     self.prepared_targets = self.prepare_targets()
        self.dataset = self.dataset.map(lambda x: self.batch_preprocess(x, **self.dataset_config), batched=True,
                            batch_size=1024, num_proc=4,
                            drop_last_batch=False, load_from_cache_file=False)
        # else:
        #     self.dataset = self.dataset.map(lambda x: self.batch_preprocess_bm(x, **dataset_config), batched=True,
        #                         batch_size=256, num_proc=4,
        #                         drop_last_batch=False, load_from_cache_file=False)
        self.dataset = self.dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
        self.candidate_dataset = self.generate_cand_dataset()
        return self.dataset, self.candidate_dataset

    def prepare_targets(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `prepare_targets` method. "
            "Please implement it in the subclass."
        )
    
    def batch_preprocess(self, batch_dict, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `batch_preprocess` method. "
            "Please implement it in the subclass."
        )

    def generate_cand_dataset(self):
        """
        Used for generating candidate datasets.
        Flatten candidates, merge with corpus, deduplication
        """
        cand_rows = []
        all_cand_name = set()
        for row in self.dataset:
            assert len(row["cand_text"]) == len(row["cand_image"]) == len(row["dataset_infos"]["cand_names"])
            for cand_text, cand_image, cand_name in zip(row["cand_text"], row["cand_image"], row["dataset_infos"]["cand_names"]):

                if cand_name not in all_cand_name:
                    cand_rows.append({
                        "cand_text": [cand_text],
                        "cand_image": [cand_image],
                        "dataset_infos": {"cand_name": cand_name},
                    })
                    all_cand_name.add(cand_name)

        if self.corpus is not None:
            for row in self.corpus:
                assert len(row["cand_text"]) == len(row["cand_image"]) == len(row["dataset_infos"]["cand_names"]) == 1
                cand_name = row["dataset_infos"]["cand_names"][0]
                if cand_name not in all_cand_name:
                    cand_rows.append({
                        "cand_text": row["cand_text"],
                        "cand_image": row["cand_image"],
                        "dataset_infos": {"cand_name": row["dataset_infos"]["cand_names"][0]},
                    })
                    all_cand_name.add(cand_name)

        cand_dataset = Dataset.from_list(cand_rows)
        return cand_dataset

    def format_text_for_chat_template(self, is_query, text, image_path=None, video_path=None, add_generation_prompt=False, key=None):

        text = text.replace(VLM_VIDEO_TOKENS[self.dataset_config["model_backbone"]], "").replace(VLM_IMAGE_TOKENS[self.dataset_config["model_backbone"]], "")

        desc = self.query_descriptions if is_query else self.target_descriptions

        description = format_description(desc[key], self.data_args.use_cot) if (desc is not None and key is not None) else ""

        formatted_sample = [
            {"role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],}
        ]
        user_content = [] 
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        if video_path:
            user_content.append({"type": "video", "video": video_path})
        user_content.append({"type": "text", "text": text})
        formatted_sample.append({"role": "user", "content": user_content})

        if not add_generation_prompt:
            formatted_sample.append({
                "role": "assistant",
                "content": [{"type": "text", "text": description if description is not None else ""}],
            })
        
        formatted_sample = self.processor.apply_chat_template(formatted_sample, add_generation_prompt=add_generation_prompt, tokenize=False)
        if not add_generation_prompt:
            formatted_sample = formatted_sample.strip()

        if self.meta_queries:
            formatted_sample += self.meta_queries
        return formatted_sample 


class MMEBEvalDatasetProcessor(BaseEvalDatasetProcessor):
    """
    MMEB evaluation dataset processor.
    It processes the dataset for MMEB evaluation tasks, including query and target descriptions.
    """

    def __init__(self, *args, 
                 query_key_text="qry_text", query_key_mm="qry_img_path",
                 cand_key_text="tgt_text", cand_key_mm="tgt_img_path",
                 **kwargs):
        super().__init__(*args, 
                        query_key_text=query_key_text, query_key_mm=query_key_mm,
                        cand_key_text=cand_key_text, cand_key_mm=cand_key_mm, 
                        **kwargs)
        self.data_parser_name = "mmeb_eval"
        self.subset_name = self.dataset_config.get("dataset_name")
        self.dataset_split = self.dataset_config.get("dataset_split", "test")

        # for MMEB v2, the query text doesn't include instructions, so we need to take the instruction part out from the description
        if self.query_descriptions is not None:
            self.query_descriptions = {
                (extract_query(qry_text, self.subset_name), qry_image_path): desc \
                for (qry_text, qry_image_path), desc in self.query_descriptions.items()
            }
        if self.target_descriptions is not None:
            self.target_descriptions = {
                (extract_target(tgt_text, self.subset_name), tgt_image_path): desc \
                for (tgt_text, tgt_image_path), desc in self.target_descriptions.items()
            }

        """
            Precompute targets to avoid repetitive processing 1000x for each sample.
        """

    # def prepare_targets(self):
    #     unique_pairs = set()
    #     for row in self.dataset:
    #         assert len(row["cand_key_text"]) == len(row["cand_key_mm"])
    #         for cand_text, cand_image in zip(row["cand_key_text"], row["cand_key_mm"]):
    #             unique_pairs.add((cand_text, cand_image))

    #     preprocessed_pairs = {}

    #     for cand_text, cand_image in unique_pairs:

    #         description = None
    #         if self.target_descriptions is not None:
    #             description = format_description(self.target_descriptions[(cand_text, cand_image)], self.data_args.use_cot)

    #         cand_text_processed = get_target(self.subset_name, cand_text, self.data_args.use_cot)
    #         cand_text_processed = self.format_text_for_chat_template(cand_text_processed, cand_image, description=description, add_generation_prompt=self.model_args.do_sft_target)
            
    #         preprocessed_pairs[(cand_text, cand_image)] = cand_text_processed
        
    #     return preprocessed_pairs

    def _load_hf_dataset(self):
                # dataset and corpus, if available
        if self.dataset_name in IMAGE_TASKS:
            dataset_path_key = "IMAGE_TASKS"  
            load_subset_name = self.dataset_name
        else:
            dataset_path_key = self.dataset_name
            load_subset_name = None

        repo_subset_split = EVAL_DATASET_HF_PATH[dataset_path_key]
        self.repo_subset_split = repo_subset_split
        return load_hf_dataset(self.repo_subset_split, subset_name=load_subset_name), None

    # def _add_signature_columns_map_func(self, batch_dict):
    #     signature_columns = {

    #         # @xuanming we assume modality in the order of text, image, video
    #         # current assume two modalities max for query and target
    #         "query_key_text": batch_dict['qry_text'],
    #         "query_key_mm": batch_dict['qry_img_path'],
    #         "cand_key_text": batch_dict['tgt_text'],
    #         "cand_key_mm": batch_dict['tgt_img_path']}
    #     return batch_dict | signature_columns

    @add_metainfo_hook
    def batch_preprocess(self, batch_dict, **kwargs):
        image_resolution, model_backbone = self.dataset_config['image_resolution'], self.dataset_config['model_backbone']
        image_root = self.dataset_config['image_root']

        query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
        for qry_text, qry_image_path, tgt_texts, tgt_image_paths in (
                zip(batch_dict['qry_text'], batch_dict['qry_img_path'], batch_dict['tgt_text'], batch_dict['tgt_img_path'])):

            qry_description = None
            if self.query_descriptions is not None:
                qry_description = format_description(self.query_descriptions[(qry_text, qry_image_path)], self.data_args.use_cot)

            if qry_image_path.strip():
                query_images.append([{"bytes": [None], "paths": [os.path.join(image_root, qry_image_path)],
                                    "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}])
            else:
                query_images.append([None])

            if not self.apply_chat_template:
                if model_backbone != PHI3V:
                    qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])

                if self.model_backbone != PHI3V:
                    qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.model_backbone])
                    tgts = []
                    for tgt_text, tgt_image_path in zip(tgt_texts, tgt_image_paths):
                        if (tgt_text, tgt_image_path) not in self.target_cache:
                            self.target_cache[(tgt_text, tgt_image_path)] = tgt_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.model_backbone])
                        tgts.append(self.target_cache[(tgt_text, tgt_image_path)])
                    cand_texts.append(tgts)
            else:
                qry_text = get_query(self.subset_name, qry_text, self.data_args.use_cot)
                qry_text = self.format_text_for_chat_template(True, 
                                                              text=qry_text, 
                                                              image_path=qry_image_path, 
                                                              add_generation_prompt=self.model_args.do_sft_query,
                                                              key=(qry_text, qry_image_path))
                cand_text = []
                for tgt_cap, tgt_img_path in zip(tgt_texts, tgt_image_paths):
                    if (tgt_cap, tgt_img_path) not in self.target_cache:
                        self.target_cache[(tgt_cap, tgt_img_path)] = self.format_text_for_chat_template(
                                                                        False, 
                                                                        text=get_target(self.subset_name, tgt_cap, self.data_args.use_cot), 
                                                                        image_path=tgt_img_path, 
                                                                        add_generation_prompt=self.model_args.do_sft_target,
                                                                        key=(tgt_cap, tgt_img_path))
                        
                    cand_text.append(self.target_cache[(tgt_cap, tgt_img_path)])
                cand_texts.append(cand_text)

            query_texts.append([qry_text])

            if tgt_image_paths[0].strip():
                cand_img_paths = [os.path.join(image_root, tgt_img_path) for tgt_img_path in tgt_image_paths]
                img_list = [{"bytes": [None], "paths": [cand_img_path],
                            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for cand_img_path in cand_img_paths]
                cand_images.append(img_list)
            else:
                cand_images.append([None] * len(tgt_texts))
            # this is used for dedup, especially important for RefCOCO-Matching, as multiple objects in the same image can be targets, so we need to use path+caption as key
            cand_names = [path+':'+cap.strip('"') for path, cap in zip(tgt_image_paths, tgt_texts)]
            dataset_infos.append({
                "cand_names": cand_names,
                "label_name": cand_names[0],
            })

        processed_batch = {"query_text": query_texts, "query_image": query_images,
                "cand_text": cand_texts, "cand_image": cand_images,
                "dataset_infos": dataset_infos}
        return batch_dict | processed_batch



class MMEBV2EvalDatasetProcessor(BaseEvalDatasetProcessor):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @add_metainfo_hook
    def batch_preprocess(self, batch_dict, *args, **kwargs):

        query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
        batch_size = len(next(iter(batch_dict.values())))

        for data_idx in range(batch_size):
            one_sample = self._process_one_sample(data_idx, batch_dict, *args, **kwargs)
            query_text, query_image, cand_text, cand_image, dataset_info = \
                one_sample['query_text'], one_sample['query_image'], \
                one_sample['cand_text'], one_sample['cand_image'], \
                one_sample['dataset_infos']

            if self.data_args.apply_chat_template:

                query_image_input = query_video_input = None
                
                if query_image:
                    if len(query_image['paths']) > 1:
                        query_video_input = query_image['paths'][0] or query_image['bytes'][0]
                    else:
                        query_image_input = query_image['paths'][0] or query_image['bytes'][0]
                
                query_key_text = batch_dict[self.query_key_text][data_idx] if self.query_key_text else ""
                query_key_mm = batch_dict[self.query_key_mm][data_idx] if self.query_key_mm else ""

                query_text = self.format_text_for_chat_template(
                                    is_query=True, 
                                    text=query_text, 
                                    image_path=query_image_input, 
                                    video_path=query_video_input, 
                                    key=(query_key_text, query_key_mm))
                
                cands = []
                for single_cand_text, single_cand_image in zip(cand_text, cand_image):
                    cand_image_input = cand_video_input = None
                    if single_cand_image:
                        if len(single_cand_image['paths']) > 1:
                            cand_video_input = single_cand_image['paths'][0] or single_cand_image['bytes'][0]
                        else:
                            cand_image_input = single_cand_image['paths'][0] or single_cand_image['bytes'][0]
                    
                    single_cand_key_mm = cand_image_input or cand_video_input
                    if (single_cand_text, single_cand_key_mm) not in self.target_cache:
                        self.target_cache[(single_cand_text, single_cand_key_mm)] = self.format_text_for_chat_template(
                                                                                        False, 
                                                                                        text=single_cand_text,
                                                                                        image_path=cand_image_input,
                                                                                        video_path=cand_video_input, 
                                                                                        add_generation_prompt=self.model_args.do_sft_target,
                                                                                        key=(single_cand_text, single_cand_key_mm))
                    cands.append(self.target_cache[(single_cand_text, single_cand_key_mm)])
                cand_text = cands

            query_texts.append([query_text])
            query_images.append([query_image])
            cand_texts.append(cand_text)
            cand_images.append(cand_image)
            dataset_infos.append(dataset_info)

        return {"query_text": query_texts, "query_image": query_images,
                "cand_text": cand_texts, "cand_image": cand_images,
                "dataset_infos": dataset_infos}