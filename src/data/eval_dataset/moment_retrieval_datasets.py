import os

from datasets import load_dataset
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import RESOLUTION_MAPPING, MMEBV2EvalDatasetProcessor
from src.data.utils.dataset_utils import load_hf_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames, VID_EXTENSIONS
from src.model.processor import process_input_text
from ..prompts import VIDEO_EMBED_INSTRUCTION
from ..loader.mixed_dataset import AutoPairEvalDataset

TASK_INST_QRY =  "Find the clip that corresponds to the described scene in the given video:"
TASK_INST_TGT =  "Understand the content of the provided video."

# TASK_INST_QRY = ""
# TASK_INST_TGT = ""

DATASET_PARSER_NAME = "moment_retrieval"
@AutoPairEvalDataset.register(DATASET_PARSER_NAME)
@AutoPairEvalDataset.register_instruction(["QVHighlight", "Charades-STA"],
    {'query': """Given a video and a query, identify the clip in the video that best matches the query.\n\nQuery: {text}\n\nEmbed the clip with your answer.""",
     'target': VIDEO_EMBED_INSTRUCTION})

class MomentRetrievalEvalDatasetProcessor(MMEBV2EvalDatasetProcessor):
    def __init__(self, *args, **dataset_config):

        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

    def _load_hf_dataset(self):
        if self.dataset_config.get("data_path", None) != None:
            dataset = load_dataset("json", data_files=self.dataset_config["data_path"])
            dataset = dataset["train"]
        else:
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[self.dataset_config['dataset_name']])
        return dataset, None

    def _process_one_sample(self, data_idx, batch_dict, *args, **kwargs):
        image_resolution       = kwargs["image_resolution"]
        max_video_frames_saved = kwargs["max_video_frames_saved"]
        max_clip_frames_saved  = kwargs["max_clip_frames_saved"]
        num_video_frames       = kwargs["num_video_frames"]
        num_clip_frames        = kwargs["num_clip_frames"]
        model_backbone         = kwargs["model_backbone"]
        video_root             = kwargs.get("video_root")
        clip_root              = kwargs.get("clip_root")
        frame_root             = kwargs.get("frame_root")

        # Pull this sample's fields
        query_text_raw   = batch_dict["query"][data_idx]
        query_video_rel  = batch_dict["video_path"][data_idx]

        video_basename = os.path.basename(query_video_rel)
        video_name     = os.path.splitext(video_basename)[0]
        frames_dir     = os.path.join(frame_root, video_name)

        # Prepare query video frames
        query_video_path = os.path.join(video_root, video_basename) if video_root else None
        query_frame_dir  = os.path.join(frames_dir, "query")

        dataset_infos = {}
        query_image   = None
        cand_text     = []
        cand_image    = []
        cand_clip_names = []
        pos_clip_name   = None

        try:
            # Save and load query frames (only extract if not already there)
            if not os.path.exists(query_frame_dir):
                save_frames(
                    video_path=query_video_path,
                    frame_dir=query_frame_dir,
                    max_frames_saved=max_video_frames_saved
                )
            qry_frame_paths = process_video_frames(query_frame_dir, num_frames=num_video_frames)

            if qry_frame_paths:
                query_image = {
                    "bytes": [None] * len(qry_frame_paths),
                    "paths": qry_frame_paths,
                    "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_frame_paths),
                }

            # If frames for candidates are not yet extracted, do so from raw clip videos
            if not os.path.exists(frames_dir):
                clip_video_dir = os.path.join(clip_root, video_name) if clip_root else None
                if clip_video_dir and os.path.isdir(clip_video_dir):
                    clip_video_paths = [
                        f for f in os.listdir(clip_video_dir)
                        if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
                    ]
                    for clip_video_file in clip_video_paths:
                        clip_name = os.path.splitext(clip_video_file)[0]
                        clip_frame_dir = os.path.join(frames_dir, clip_name)
                        save_frames(
                            video_path=os.path.join(clip_video_dir, clip_video_file),
                            frame_dir=clip_frame_dir,
                            max_frames_saved=max_clip_frames_saved
                        )

            target_description = []
            # Collect candidate clips from frames_dir
            if os.path.exists(frames_dir):
                for entry in os.listdir(frames_dir):
                    clip_dir_abs = os.path.join(frames_dir, entry)
                    # skip the query folder and any regular files at this level
                    if entry == "query" or os.path.isfile(clip_dir_abs):
                        continue

                    if entry.startswith("positive"):
                        pos_clip_name = clip_dir_abs

                    frame_paths = process_video_frames(clip_dir_abs, num_frames=num_clip_frames)
                    cand_image.append({
                        "bytes": [None] * len(frame_paths),
                        "paths": frame_paths,
                        "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)] * len(frame_paths),
                    })
                    cand_clip_names.append(clip_dir_abs) 
                    if self.target_descriptions:
                        target_desc = self.target_descriptions.get((os.path.join(video_name, entry),))
                        target_description.append(target_desc)
                        # if not target_desc:
                        #     print(f'No target description found for ({os.path.join(video_name, entry)},) for dataset {DATASET_PARSER_NAME}')
                    else:
                        target_description.append(None)

                # Text for each candidate (parent will handle chat-template if enabled)
                if len(cand_clip_names) > 0:
                    tgt_txt = process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)
                    cand_text = [tgt_txt] * len(cand_clip_names)

            dataset_infos = {
                "cand_names": cand_clip_names,
                "label_name": pos_clip_name,
            }

        except Exception as e:
            dataset_infos = {
                "cand_names": cand_clip_names,
                "label_name": pos_clip_name,
                "error": str(e),
            }

        query_text = process_input_text(TASK_INST_QRY, model_backbone, text=query_text_raw, add_video_token=True)
        query_description = None
        if self.query_descriptions:
            query_description = self.query_descriptions.get((query_text_raw, video_name))
            if not query_description:
                print(f'No query description found for ({query_text_raw}, {video_name}) for dataset {self.dataset_config["dataset_name"]}')

        return {
            "query_text": query_text,
            "query_image": query_image,   # dict with paths/bytes/resolutions or None
            "cand_text": cand_text,       # list[str], len == num candidates
            "cand_image": cand_image,     # list[dict], aligned with cand_text
            "dataset_infos": dataset_infos,
            "query_description": query_description,
            "target_description": target_description,  # list[str] for each candidate
        }