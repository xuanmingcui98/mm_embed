# Video Classification
from .video_classification_datasets import VideoClassificationEvalDatasetProcessor
from .ssv2_dataset import SSV2EvalDatasetProcessor

# Video QA
from .videomme_dataset import VideoMMMEvalDatasetProcessor
from .mvbench_dataset import MVBenchEvalDatasetProcessor
from .nextqa_dataset import NextQAEvalDatasetProcessor
from .egoschema_dataset import EgoSchemaEvalDatasetProcessor
from .activitynetqa_dataset import ActivityNetQAEvalDatasetProcessor
from .videommmu_dataset import VideoMMMUEvalDatasetProcessor

# Video Retrieval
from .msrvtt_dataset import MSRVTTEvalDatasetProcessor
from .didemo_dataset import DiDemoEvalDatasetProcessor
from .msvd_dataset import MSVDEvalDatasetProcessor
from .youcook2_dataset import YouCook2EvalDatasetProcessor
from .vatex_dataset import VatexEvalDatasetProcessor

from .gui_dataset import GuiEvalDatasetProcessor

# Temporal Grounding
from .moment_retrieval_datasets import MomentRetrievalEvalDatasetProcessor
from .momentseeker_dataset import MomentSeekerEvalDatasetProcessor

# MMEB
# from .image_cls_dataset import load_image_cls_dataset
# from .image_qa_dataset import load_image_qa_dataset
# from .image_t2i_eval import load_image_t2i_dataset
# from .image_i2t_eval import load_image_i2t_dataset
# from .image_i2i_vg_dataset import load_image_i2i_vg_dataset

from .image_cls_dataset import ImageClsEvalDatasetProcessor
from .image_qa_dataset import ImageQAEvalDatasetProcessor
from .image_t2i_eval import ImageT2IEvalDatasetProcessor
from .image_i2t_eval import ImageI2TEvalDatasetProcessor
from .image_i2i_vg_dataset import ImageI2IVGEvalDatasetProcessor

# VisDoc
from .vidore_dataset import VidoreEvalDatasetProcessor
from .visrag_dataset import VisRAGEvalDatasetProcessor
