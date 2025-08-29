from datasets import load_dataset
from PIL import Image
from datasets.features.image import image_to_bytes
from torch.jit import isinstance
from ..dataset.base_pair_dataset import RESOLUTION_MAPPING, VideoDatasetProcessor
from ..loader.mixed_dataset import AutoPairDataset
from src.model.processor import VLM_IMAGE_TOKENS
from ..prompts import TEXT_EMBED_INSTRUCTION, VIDEO_EMBED_INSTRUCTION, VISDOC_QA_RETRIEVAL_INSTRUCTION, VISDOC_EMBED_INSTRUCTION

def process_query(query, prompt, image_token=''):
    if prompt:
        query = f'{prompt} {query} {image_token}'
    else:
        query = f'{query} {image_token}'
    return query

query_source2prompt = {
    "NeurIPS Papers": "This query is about a research paper from NeurIPS, a leading AI/ML conference. The document contains technical discussions, methodologies, and findings. Identify relevant papers and sections that address the query: ",  # 10,000
    "Textbooks": "This query is related to a college-level textbook, which provides structured explanations, definitions, and examples. Find the most relevant concepts or explanations that address the query: ",    # 5,000
    "ICML Papers": "This query is about a research paper from ICML, a leading AI/ML conference. The document contains theoretical insights, experiments, and applications. Identify relevant papers and sections that best answer the query: ",    # 5,000
    "Manuallib": "This query pertains to a product manual, which contains detailed technical specifications, usage instructions, and troubleshooting steps. Find the most relevant section that answers the query: ",    # 20,000
    "ArxivQA": "This query is related to retrieving a relevant figure from an ArXiv research paper. The retrieved figure should contain scientific plots, mathematical visualizations, or experimental results that best address the query: ",  # 25,856
    "ChartQA": "This query is related to retrieving a relevant chart that visually represents numerical or categorical data. The retrieved chart should contain bar graphs, line charts, or other visual elements necessary to analyze trends, compare values, or extract insights related to the query: ",	  # 4,224
    "MP-DocVQA": "This query is related to retrieving a relevant page from a multi-page document, such as reports, invoices, or research papers. The retrieved document should contain text, tables, or structured information necessary to answer the query: ",	  # 10,624
    "InfoVQA": "This query is related to retrieving an infographic that visually presents statistical or factual information using charts, icons, and structured layouts. The retrieved image should contain the necessary visual elements to provide the best context for answering the query: ",	  # 17,664
    "PlotQA": "This query relates to retrieving a relevant plot or chart that visually represents numerical data. The retrieved figure should contain the necessary information to analyze trends, compare values, or extract key insights related to the query: ",	  # 56,192
    "SlideVQA": "This query is related to retrieving a relevant presentation slide that visually presents structured information. The retrieved slide should contain the necessary text, charts, or graphics to provide the best answer to the query: ",	  # 8,192
}
target_source2prompt = {
    "Textbooks": "A textbook page with structured educational content and explanations.",
    "ICML Papers": "A research paper from ICML, covering machine learning topics.",
    "NeurIPS Papers": "A research paper from NeurIPS on AI and ML topics.",
    "Manuallib": "A product manual page with technical specifications and instructions.",
    "InfoVQA": "An infographic with structured data, charts, and annotations.",
    "PlotQA": "A numerical data visualization, such as bar charts or line graphs.",
    "SlideVQA": "A presentation slide with text, bullet points, and diagrams.",
    "ArxivQA": "A figure from a research paper, including plots or experimental results.",
    "MP-DocVQA": "A page from a multi-page document with text or tables.",
    "ChartQA": "A statistical chart comparing values or analyzing trends.",
}

TASK_INST_TGT = "Represent the following text:\n"
DATASET_PARSER_NAME = "visrag"
@AutoPairDataset.register(DATASET_PARSER_NAME)
@AutoPairDataset.register_instruction("VisRag-Indomain-data", 
    {'query': VISDOC_QA_RETRIEVAL_INSTRUCTION,
     'target': VISDOC_EMBED_INSTRUCTION})
class VisragDatasetProcessor(VideoDatasetProcessor):
    def __init__(self, *args, **dataset_config):
        super().__init__(DATASET_PARSER_NAME, *args, **dataset_config)

    def _load_hf_dataset(self):
        dataset_name = self.dataset_config.get("dataset_name", DATASET_PARSER_NAME)
        dataset_split = self.dataset_config.get("dataset_split", "train")
        dataset_path = self.dataset_config.get("dataset_path", None)

        if dataset_name:
            dataset = load_dataset("openbmb/VisRAG-Ret-Train-In-domain-data", split=dataset_split)
        elif dataset_path:
            dataset = load_dataset("parquet", data_files=dataset_path, split="train")
        
        dataset = dataset.add_column("id", list(range(len(dataset))))
        return dataset

    def _process_one_sample(self, idx, batch_dict, *args, **kwargs):
        model_backbone = kwargs['model_backbone']
        image_resolution = kwargs['image_resolution']

        query, image, source = batch_dict['query'][idx], batch_dict['image'][idx], batch_dict['source'][idx]
        # query = process_query(query, prompt=query_source2prompt.get(source, ""), image_token="")
        # pos_text = process_query('', prompt=target_source2prompt.get(source, ""), image_token=VLM_IMAGE_TOKENS[model_backbone])
        query = process_query(query, prompt="", image_token="")
        pos_text = process_query('', prompt="", image_token=VLM_IMAGE_TOKENS[model_backbone])

        if isinstance(image, Image.Image):
            # BC, datasets==2.21.0
            image_bytes = image_to_bytes(image)
            path = ""
        elif type(image) is dict:
            # datasets==3.3.2
            image_bytes = image['bytes']
            path = image['path']
        pos_image = {"bytes": [image_bytes], "paths": [path], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}


        target_description = None
        if self.target_descriptions:
            target_description = self.target_descriptions.get((batch_dict['id'][idx],))
            if target_description is None:
                print(f"No target description found for id {batch_dict['id'][idx]} for {self.dataset_config['dataset_name']} dataset")
                
        return {"query_text": query, 
                "query_image": None,
                "pos_text": pos_text, 
                "pos_image": pos_image,
                "neg_text": "", 
                "neg_image": None,
                "query_description": None,
                "target_description": target_description}
