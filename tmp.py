from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import hf_hub_download

MODEL_PATH = "facebook/Perception-LM-1B"
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH).to("cuda")