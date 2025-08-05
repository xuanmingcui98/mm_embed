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

        self.subset_name = self.dataset_config.get("dataset_name")
        self.dataset_split = self.dataset_config.get("dataset_split", "test")

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

        if self.subset_name in IMAGE_TASKS:
            dataset_path_key = "IMAGE_TASKS"  
            self.load_subset_name = self.subset_name
        else:
            dataset_path_key = self.subset_name
            self.load_subset_name = None

        repo_subset_split = EVAL_DATASET_HF_PATH[dataset_path_key]
        self.dataset_name = repo_subset_split[0]
        self.dataset = self._load_hf_dataset()
        self.dataset = sample_dataset(self.dataset, **dataset_config)

        self.corpus = self.prepare_corpus()

        self.dataset_config['model_backbone'] = model_args.model_backbone
        self.dataset_config['image_resolution'] = data_args.image_resolution

        if self.data_args.apply_chat_template:
            self.prepared_targets = self.prepare_targets()
            self.dataset = self.dataset.map(lambda x: self.batch_preprocess(x), batched=True,
                                batch_size=1024, num_proc=4,
                                drop_last_batch=False, load_from_cache_file=False)
        else:
            self.dataset = self.dataset.map(lambda x: self.batch_preprocess_bm(x, **dataset_config), batched=True,
                                batch_size=256, num_proc=4,
                                drop_last_batch=False, load_from_cache_file=False)
        self.dataset = self.dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    def _load_hf_dataset(self):
        return load_hf_dataset(self.dataset_name, subset_name=self.load_subset_name)

    def load(self):
        return self.dataset, self.corpus

    def prepare_corpus(self):
        
        return 

    def prepare_targets(self):
        """
            Precompute targets to avoid repetitive processing 1000x for each sample.
        """

        unique_pairs = set()
        for row in self.dataset:
            assert len(row["tgt_text"]) == len(row["tgt_img_path"])
            for cand_text, cand_image in zip(row["tgt_text"], row["tgt_img_path"]):
                unique_pairs.add((cand_text, cand_image))

        preprocessed_pairs = {}

        for cand_text, cand_image in unique_pairs:

            description = None
            if self.target_descriptions is not None:
                description = format_description(self.target_descriptions[(cand_text, cand_image)], self.data_args.use_cot)

            cand_text_processed = get_target(self.subset_name, cand_text, self.data_args.use_cot)
            cand_text_processed = format_text_for_chat_template(self.processor, cand_text_processed, cand_image, description=description, add_generation_prompt=self.model_args.do_sft_target) + self.meta_queries
            
            preprocessed_pairs[(cand_text, cand_image)] = cand_text_processed
        
        return preprocessed_pairs

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

            qry_text = get_query(self.subset_name, qry_text, self.data_args.use_cot)
            qry_text = format_text_for_chat_template(self.processor, qry_text, qry_image_path, description=qry_description, add_generation_prompt=self.model_args.do_sft_query) + self.meta_queries
            query_texts.append([qry_text])

            if qry_image_path.strip():
                query_images.append([{"bytes": [None], "paths": [os.path.join(image_root, qry_image_path)],
                                    "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}])
            else:
                query_images.append([None])

            cand_texts.append([self.prepared_targets[(tgt_cap, tgt_img_path)] for tgt_cap, tgt_img_path in zip(tgt_texts, tgt_image_paths)])

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

        return {"query_text": query_texts, "query_image": query_images,
                "cand_text": cand_texts, "cand_image": cand_images,
                "dataset_infos": dataset_infos}

    @add_metainfo_hook
    def batch_preprocess_bm(self, batch_dict, *args, **kwargs):
        image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
        image_root = kwargs['image_root']

        query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
        for qry_inst, qry_text, qry_image_path, tgt_captions, tgt_image_paths in (
                zip(batch_dict['qry_inst'], batch_dict['qry_text'], batch_dict['qry_img_path'], batch_dict['tgt_text'], batch_dict['tgt_img_path'])):
            

            if model_backbone != PHI3V:
                qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])

            qry_inst = "\n" + qry_inst.replace("<|image_1|>", "").strip()
            qry_text = process_input_text(qry_inst, model_backbone, text=qry_text, add_image_token=True)
            # to stay consistent with v1 eval
            qry_text = qry_text.replace(" \n", "\n") + "\n"
            query_texts.append([qry_text])
            if qry_image_path.strip():
                qry_image_path = os.path.join(image_root, qry_image_path)
                query_images.append([{"bytes": [None], "paths": [qry_image_path],
                                    "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}])
            else:
                query_images.append([None])

            # subtle target side processing, to stay consistent with v1 eval
            if tgt_captions[0].strip():  # RefCOCO-Matching has valid text inputs
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
            for tgt_cap in tgt_captions:
                    tgt_inst_captions.append(tgt_cap.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]))
            cand_texts.append(tgt_inst_captions)

            if tgt_image_paths[0].strip():
                cand_img_paths = [os.path.join(image_root, tgt_img_path) for tgt_img_path in tgt_image_paths]
                img_list = [{"bytes": [None], "paths": [cand_img_path],
                            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for cand_img_path in cand_img_paths]
                cand_images.append(img_list)
            else:
                cand_images.append([None] * len(tgt_captions))
            # this is used for dedup, especially important for RefCOCO-Matching, as multiple objects in the same image can be targets, so we need to use path+caption as key
            cand_names = [path+':'+cap.strip('"') for path, cap in zip(tgt_image_paths, tgt_captions)]
            dataset_infos.append({
                "cand_names": cand_names,
                "label_name": cand_names[0],
            })

        return {"query_text": query_texts, "query_image": query_images,
                "cand_text": cand_texts, "cand_image": cand_images,
                "dataset_infos": dataset_infos}


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



class VideoEvalDatasetProcessor(BaseEvalDatasetProcessor):
    def __init__(self, data_parser_name, model_args, data_args, training_args, processor, **dataset_config):
        super().__init__(data_parser_name, model_args, data_args, training_args, processor, **dataset_config)

        num_frames, max_frames_saved = dataset_config['num_frames'], dataset_config['max_frames_saved']
        video_root, frame_root = dataset_config['video_root'], dataset_config['frame_root']



