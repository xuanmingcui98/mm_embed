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
                       extract_query_from_mmeb, extract_target_from_mmeb)
import torch
from src.model.processor import process_input_text
from ..utils.dataset_utils import load_hf_dataset, sample_dataset
from ..dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.video_classification_utils import VIDEOCLS_LABEL_MAPPING, DATASET_INSTRUCTION
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
                 **dataset_config):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.processor = processor
        self.dataset_config = dataset_config
        self.data_parser_name = data_parser_name

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



    def _add_signature_columns_map_func(self, batch_dict):
        
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `_add_signature_columns_map_func` method. "
            "Please implement it in the subclass."
        )

    def add_signature_columns(self):

        self.dataset = self.dataset.map(
            self._add_signature_columns_map_func,
            batched=True,
            batch_size=2048,
            num_proc=4,
            drop_last_batch=False,
            load_from_cache_file=False
        )


    def _load_hf_dataset(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `_load_hf_dataset` method. "
            "Please implement it in the subclass."
        )

    def load(self):

        self.dataset, self.corpus = self._load_hf_dataset()
        self.dataset = sample_dataset(self.dataset, **self.dataset_config)

        if self.data_args.apply_chat_template:
            self.add_signature_columns()
            self.prepared_targets = self.prepare_targets()
        self.dataset = self.dataset.map(lambda x: self.batch_preprocess(x, **self.dataset_config), batched=True,
                            batch_size=1024, num_proc=4,
                            drop_last_batch=False, load_from_cache_file=False)
        # else:
        #     self.dataset = self.dataset.map(lambda x: self.batch_preprocess_bm(x, **dataset_config), batched=True,
        #                         batch_size=256, num_proc=4,
        #                         drop_last_batch=False, load_from_cache_file=False)
        self.dataset = self.dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos",
                                                    "query_key_text", "query_key_mm", "cand_key_text", "cand_key_mm"])
        self.candidate_dataset = self.generate_cand_dataset()
        return self.dataset, self.candidate_dataset

    def prepare_targets(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `prepare_targets` method. "
            "Please implement it in the subclass."
        )
    
    def batch_preprocess(self, batch_dict):
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
            for cand_text, cand_image, cand_name, cand_key_text, cand_key_mm in zip(row["cand_text"], row["cand_image"], row["dataset_infos"]["cand_names"], row["cand_key_text"], row["cand_key_mm"]):
                if self.processor.apply_chat_template:
                    cand_text = self.prepared_targets[(cand_key_text, cand_key_mm)]
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

    def format_text_for_chat_template(self, text, image_path=None, video_path=None, description=None, add_generation_prompt=False):

        if description is not None:
            description = format_description(description, self.data_args.use_cot)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_parser_name = "mmeb_eval"
        self.subset_name = self.dataset_config.get("dataset_name")
        self.dataset_split = self.dataset_config.get("dataset_split", "test")

        # for MMEB v2, the query text doesn't include instructions, so we need to take the instruction part out from the description
        if self.query_descriptions is not None:
            self.query_descriptions = {
                (extract_query_from_mmeb(qry_text, self.subset_name), qry_image_path): desc \
                for (qry_text, qry_image_path), desc in self.query_descriptions.items()
            }
        if self.target_descriptions is not None:
            self.target_descriptions = {
                (extract_target_from_mmeb(tgt_text, self.subset_name), tgt_image_path): desc \
                for (tgt_text, tgt_image_path), desc in self.target_descriptions.items()
            }

        """
            Precompute targets to avoid repetitive processing 1000x for each sample.
        """

    def prepare_targets(self):
        unique_pairs = set()
        for row in self.dataset:
            assert len(row["cand_key_text"]) == len(row["cand_key_mm"])
            for cand_text, cand_image in zip(row["cand_key_text"], row["cand_key_mm"]):
                unique_pairs.add((cand_text, cand_image))

        preprocessed_pairs = {}

        for cand_text, cand_image in unique_pairs:

            description = None
            if self.target_descriptions is not None:
                description = format_description(self.target_descriptions[(cand_text, cand_image)], self.data_args.use_cot)

            cand_text_processed = get_target(self.subset_name, cand_text, self.data_args.use_cot)
            cand_text_processed = self.format_text_for_chat_template(cand_text_processed, cand_image, description=description, add_generation_prompt=self.model_args.do_sft_target)
            
            preprocessed_pairs[(cand_text, cand_image)] = cand_text_processed
        
        return preprocessed_pairs

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

    def _add_signature_columns_map_func(self, batch_dict):
        signature_columns = {

            # @xuanming we assume modality in the order of text, image, video
            # current assume two modalities max for query and target
            "query_key_text": batch_dict['qry_text'],
            "query_key_mm": batch_dict['qry_img_path'],
            "cand_key_text": batch_dict['tgt_text'],
            "cand_key_mm": batch_dict['tgt_img_path']}
        return batch_dict | signature_columns

    @add_metainfo_hook
    def batch_preprocess(self, batch_dict):
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

                qry_inst = "\n" + qry_inst.replace("<|image_1|>", "").strip()
                qry_text = process_input_text(qry_inst, model_backbone, text=qry_text, add_image_token=True)
                # to stay consistent with v1 eval
                qry_text = qry_text.replace(" \n", "\n") + "\n"


                if tgt_texts[0].strip():  # RefCOCO-Matching has valid text inputs
                    tgt_inst = tgt_inst.replace("<|image_1|>", "")
                    tgt_inst_captions = []
                    
                    tgt_inst_caption = process_input_text(tgt_inst + ' ' + tgt_cap, model_backbone, text='', add_image_token=True)
                    tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n") + '\n'
                    tgt_inst_captions.append(tgt_inst_caption)
                    cand_texts.append(tgt_inst_captions)
                else:
                    tgt_inst = tgt_inst.replace("<|image_1|>", "")
                    tgt_inst_caption = process_input_text(tgt_inst, model_backbone, text='', add_image_token=True)
                    tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n")  # to stay consistent with v1 eval

                    cand_texts.append([tgt_inst_caption] * len(tgt_image_paths))
                tgt_inst_captions = []
                for tgt_cap in tgt_texts:
                        tgt_inst_captions.append(tgt_cap.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]))

            else:
                qry_text = get_query(self.subset_name, qry_text, self.data_args.use_cot)
                qry_text = self.format_text_for_chat_template(qry_text, qry_image_path, description=qry_description, add_generation_prompt=self.model_args.do_sft_query)

                cand_texts.append([self.prepared_targets[(tgt_cap, tgt_img_path)] for tgt_cap, tgt_img_path in zip(tgt_texts, tgt_image_paths)])

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


    def __init__(self, *args, query_instruction=None, target_instruction=None, target_modality="video", **kwargs):
        super().__init__(*args, **kwargs)
        self.query_instruction = query_instruction
        self.target_instruction = target_instruction
        self.target_modality = target_modality

    def prepare_targets(self):
        """
            Precompute targets to avoid repetitive processing 1000x for each sample.
        """

        unique_pairs = set()
        for row in self.dataset:
            if not isinstance(row["cand_key_text"], list):
                unique_pairs.add((row["cand_key_text"], row["cand_key_mm"]))
            else:
                assert len(row["cand_key_text"]) == len(row["cand_key_mm"])
                for cand_text, cand_mm in zip(row["cand_key_text"], row["cand_key_mm"]):
                    unique_pairs.add((cand_text, cand_mm))

        preprocessed_pairs = {}

        for cand_text, cand_mm in unique_pairs:

            description = self.target_descriptions[(cand_text, cand_mm)] if self.target_descriptions is not None else None

            cand_text_processed = process_input_text(self.target_instruction, self.model_backbone, text=cand_text)
            input_kwargs = {
                "text": cand_text_processed,
                "description": description,
                "add_generation_prompt": self.model_args.do_sft_target
            }

            if self.target_modality == "video":
                input_kwargs["video_path"] = cand_mm
            elif self.target_modality == "text":
                input_kwargs["image_path"] = cand_mm

            preprocessed_pairs[(cand_text, cand_mm)] = self.format_text_for_chat_template(**input_kwargs)
        
        return preprocessed_pairs

