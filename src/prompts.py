tasks = ['ImageNet-1K', "ImageNet_1K", 'N24News', 'HatefulMemes', 'VOC2007', 'SUN397', 'A-OKVQA', 'MSCOCO', 
         'Place365', 'ImageNet-A', 'ImageNet-R', 'ObjectNet', 'Country211', 'OK-VQA', 'RefCOCO', 
         'DocVQA', 'InfographicsVQA', 'ChartQA', 'NIGHTS', 'FashionIQ', 'ScienceQA', 'Visual7W', 
         'VizWiz', 'GQA', 'TextVQA', 'VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 
         'MSCOCO_i2t', 'Wiki-SS-NQ', 'WebQA', 'OVEN', 'EDIS', 'RefCOCO-Matching', 'Visual7W-Pointing']

task_categories = {
    "classification": {"ImageNet-1K", "ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"},
    "vqa": {"OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"},
    "retrieval": {"VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"},
    "grounding": {"MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"}
}


query_user_prompts_base = {
# vqa
    "OK-VQA":  """Question: {query}

Answer the above question in a single word or phrase based on the given image.""",
    "A-OKVQA": """Question: {query}

Answer the above question in a single word or phrase based on the given image.""",
    "DocVQA": """Question: {query}

Answer the above question in a single word or phrase based on the given image.""",
    "InfographicsVQA": """Question: {query}

Answer the above question in a single word or phrase based on the given image.""",
    "ScienceQA": """Question: {query}

Answer the above question in a single word or phrase based on the given image.""",
    "Visual7W": """Question: {query}

Answer the above question in a single word or phrase based on the given image.""",
    "VizWiz": """Question: {query}

Answer the above question in a single word or phrase based on the given image.""",
    "GQA": """Question: {query}

Answer the above question in a single word or phrase based on the given image.""",
    "TextVQA": """Question: {query}

Answer the above question in a single word or phrase based on the given image.""",
    "ChartQA": """Question: {query}

Answer the above question in a single word or phrase based on the given chart image.""",

# classification
    "HatefulMemes": """Identify the text in the image. Does it contain hateful speech or not? Answer with "Yes" or "No".""",
    "VOC2007": """What is the main object category of the image? Answer in a single word or phrase.""",
    "SUN397": """What is the scene in the provided image? Answer in a single word or phrase.""",
    "Place365": """What is the scene in the provided image? Answer in a single word or phrase.""",
    "ImageNet-1K": """What is the main object category of the image? Answer in a single word or phrase.""",
    "ImageNet_1K": """What is the main object category of the image? Answer in a single word or phrase.""",
    "ImageNet-A": """What is the main object category of the image? Answer in a single word or phrase.""",
    "ImageNet-R": """What is the main object category of the image? Answer in a single word or phrase.""",
    "ObjectNet": """What is the main object category of the image? Answer in a single word or phrase.""",
    "Country211": """What is the country where the image was taken? Answer in a single word or phrase.""",
    "N24News": """What is the main domain of the news image and its text below?

News text: {query}

Answer with a single word or short phrase.""",

# retrieval

    "CIRR": """Modification instruction: {query}

Describe the target image based on the base image and the modification instruction.""",
    "FashionIQ": """Modification instruction: {query}

Describe the target fashion image based on the base fashion image and the modification instruction.""",

    "VisDial": """Dialogue: {query}

Generate a description of the image based on the dialogue.""",
    "VisualNews_i2t": """Generate a concise and succinct COCO-style caption of the image as similar as possible to the target caption.""",
    "WebQA": """Given the following question, generate a description of the target retrieval text-image pair which can answer the question.
      
Question: {query}""",
    "EDIS": """Given the following news caption, generate a description of what the target news text-image pair may look like based on the caption.
    
News caption: {query}""",
    "Wiki-SS-NQ": """Generate a description of the target retrieval document screenshot that can answer the below question.
    
Question: {query}""",
    "OVEN": """Generate a description of the target retrieval text-image pair which can answer the below visual question.
    
Question: {query}""",
    "VisualNews_t2i": """News text: {query}

Generate a more concise and detailed description of the target news image based on the provided news text.""",
    "MSCOCO_t2i": """Caption: {query}

Generate a more detailed and concise description of the target image based on the provided COCO-style caption.""",
    "MSCOCO_i2t": """Describe what the target image may look like based on the given image.""",

    "NIGHTS": """Describe what the target image may look like based on the given image.""",

    # grounding

    "MSCOCO": """Describe the {query} in the image in detail.""",
    "Visual7W-Pointing": """Based on the provided image, identify the object or region the below question refers to, and provide a concise description of the object or region.

Question: {query}""",
    "RefCOCO": """Based on the provided image, identify the object or region the below expression refers to, and provide a concise description of the object or region.

Expression: {query}""",
    "RefCOCO-Matching": """Based on the provided image, identify the object or region the below expression refers to, and provide a concise description of the object or region.

Expression: {query}""",

}

query_user_prompts_cl = {}

for k, v in query_user_prompts_base.items():
    if k in task_categories['vqa'] or k in task_categories['classification'] or k in {"VisualNews_i2t", "MSCOCO_i2t"}:
        query_user_prompts_cl[k] = v + "\n\nFinally, represent your answer as text embeddings."
    elif k in task_categories['grounding'] or k in {"VisDial", "VisualNews_t2i", "MSCOCO_t2i", "CIRR", "FashionIQ", "NIGHTS"}:
        query_user_prompts_cl[k] = v + "\n\nFinally, represent the target image as image embeddings."
    elif k in {"WebQA", "EDIS", "OVEN"}:
        query_user_prompts_cl[k] = v + "\n\nFinally, represent the target as image-text multimodal embeddings."
    elif k in {"Wiki-SS-NQ"}:
        query_user_prompts_cl[k] = v + "\n\nFinally, represent the target as document embeddings."
    else:
        print(f"Task {k} is not categorized correctly or missing in query_user_prompts_base.")

query_system_prompts_base = {
# VQA tasks
    "OK-VQA": """You are a Vision Language Model specialized in answering questions based on daily images.
Your task is to analyze the provided image and respond to queries with concise answers as a single word or short phrase.""",
    "A-OKVQA": """You are a Vision Language Model specialized in answering visual reasoning questions based on daily images.
Your task is to analyze the provided image and respond to queries with concise answers as a single word or short phrase.""",
    "DocVQA": """You are a Vision Language Model specialized in answering questions based on document images.
Your task is to analyze and understand the content of the document image and respond to queries with concise answers as a single word or short phrase.""",
    "InfographicsVQA": """You are a Vision Language Model specialized in answering questions based on infographics images.
Your task is to analyze the provided infographic image and respond to queries with concise answers as a single word or short phrase.""",
    "ChartQA": """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze charts and texts in the provided chart image and respond to queries with concise answers as a single word, number, or short phrase.""",
    "ScienceQA": """You are a Vision Language Model specialized in answering questions based on scientific images.
Your task is to analyze the provided scientific image and respond to queries with concise answers as a single word or short phrase.""",
    "Visual7W": """You are a Vision Language Model specialized in answering questions about images.
Your task is to analyze the provided image and respond to queries with concise answers as a single word or short phrase.""",
    "VizWiz": """You are a Vision Language Model specialized in answering questions based on images.
Your task is to analyze the provided image and respond to queries with concise answers as a single word or short phrase.""",
    "GQA": """You are a Vision Language Model specialized in answering questions based on images.
Your task is to analyze the provided image and respond to queries with concise answers as a single word or short phrase.""",
    "TextVQA": """You are a Vision Language Model specialized in answering questions based on images with text.
Your task is to analyze the provided image and respond to queries with concise answers as a single word or short phrase.""",

# Classification tasks
    "ImageNet-1K": """You are a Vision Language Model specialized in image classification.
Your task is to analyze the provided image and identify the main object category as a single word or short phrase.""",
    "ImageNet_1K": """You are a Vision Language Model specialized in image classification.
Your task is to analyze the provided image and identify the main object category as a single word or short phrase.""",
    "ImageNet-A": """You are a Vision Language Model specialized in image classification.
Your task is to analyze the provided image and identify the main object category as a single word or short phrase.""",
    "ImageNet-R": """You are a Vision Language Model specialized in image classification.
Your task is to analyze the provided image and identify the main object category as a single word or short phrase.""",
    "N24News": """You are a Vision Language Model specialized in domaind classification.
Your task is to analyze the provided news text-image pair and identify the main domain of the news by answering with a single word or short phrase.""",
    "VOC2007": """You are a Vision Language Model specialized in image classification.
Your task is to analyze the provided image and identify the main object category by answering with a single word or short phrase.""",
    "SUN397": """You are a Vision Language Model specialized in scene classification.
Your task is to analyze the provided image and identify the scene of the image by answering with a single word or short phrase.""",
    "ObjectNet": """You are a Vision Language Model specialized in object classification.
Your task is to analyze the provided image and identify the main object category by answering with a single word or short phrase.""",
    "Country211": """You are a Vision Language Model specialized in country classification.
Your task is to analyze the provided image and identify the country where the image was taken by answering with a single word or short phrase.""",
    "HatefulMemes": """You are a Vision Language Model specialized in detecting hateful speech in images.
Your task is to identify the text in the image and determine if it contains hateful speech or not by answering "Yes" or "No".""",
    "Place365": """You are a Vision Language Model specialized in scene classification.
Your task is to analyze the provided image and identify the scene of the image as a single word or short phrase.""",

# Retrieval tasks

    "VisDial": """You are a Vision Language Model specialized in representing images based on dialogues.
Your task is to analyze the provided dialogue about an image and generate a concise description of the image based on the dialogue.""",
    "VisualNews_t2i": """You are a Vision Language Model specialized in representing news images based on provided news text.
Your task is to analyze the provided news text, and generate a more concise and detailed description of the target news image.""",
    "MSCOCO_i2t": """You are a Vision Language Model specialized in COCO-style image-to-text retrieval.
Your task is to analyze the provided image and generate a concise, succinct COCO-style caption as similar as possible to the target caption about this image.""",
    "MSCOCO_t2i": """You are a Vision Language Model specialized in COCO-style text-to-image retrieval.
Your task is to analyze the provided COCO-style caption and generate a more detailed and concise description of the target image based on the caption.""",
    "WebQA": """You are a Vision Language Model specialized in retrieving relevant text-image pairs to answer the provided question.
Your task is to analyze the provided question and generate a description of the target retrieval text-image pair which can answer the question.""",
    "EDIS": """You are a Vision Language Model specialized in retrieving relevant text-image pairs given the provided news caption.
Your task is to analyze the provided news caption and generate a description of the target retrieval text-image.""",
    "Wiki-SS-NQ": """You are a Vision Language Model specialized in document screenshots retrieval.
Your task is to analyze the provided question and generate a description of the target retrieval document screenshot.""",
    "VisualNews_i2t": """You are a Vision Language Model specialized in news image-to-text retrieval.
Your task is to analyze the provided news image and generate a concise and succinct COCO-style caption of the image as similar as possible to the target caption.""",
    "CIRR": """You are a Vision Language Model specialized in composed image retrieval.
Your task is to analyze the provided base image and a modification instruction, which describes the changes to be applied to the base image in order to get a target image, and generate a description of the target image based on the base image and the modification instruction.""",
    "FashionIQ": """You are a Vision Language Model specialized in composed image retrieval in fashion domain.
Your task is to analyze the provided base fashion image and a modification instruction which describes the changes to be applied to the base image in order to get a target fashion image, and generate a description of the target fashion image based on the base image and the modification instruction.""",
    "NIGHTS": """You are a Vision Language Model specialized in image-to-image retrieval.
Your task is to analyze the provided base image and generate a concise description of what the target image may look like based on the base image.""",
    "OVEN": """You are a Vision Language Model specialized in text-image retrieval for visual question answering.
Your task is to analyze the provided image and question, and generate a description of the target retrieval text-image pair which can answer the given visual question.""",

# Grounding tasks
    "MSCOCO": """You are a Vision Language Model specialized in object grounding.
Your task is to analyze the provided query and locate the object or region that the query refers to in the image, and generate a concise description of the object or region.""",
    "Visual7W-Pointing": """You are a Vision Language Model specialized in object grounding.
Your task is to analyze the provided query and locate the object or region that the query refers to in the image, and generate a concise description of the object or region.""",
    "RefCOCO": """You are a Vision Language Model specialized in object grounding.
Your task is to analyze the provided query and locate the object or region that the query refers to in the image, and generate a concise description of the object or region.""",
    "RefCOCO-Matching": """You are a Vision Language Model specialized in object grounding.
Your task is to analyze the provided query and locate the object or region that the query refers to in the image, and generate a concise description of the object or region.""",
}

query_system_prompts_cl = {}

for k,v in query_system_prompts_base.items():
    if k in task_categories['vqa'] or k in task_categories['classification'] or k in {"VisualNews_i2t", "MSCOCO_i2t"}:
        query_system_prompts_cl[k] = v + "\n\nFinally, represent your answer as text embeddings."
    elif k in task_categories['grounding'] or k in {"VisDial", "VisualNews_t2i", "MSCOCO_t2i", "CIRR", "FashionIQ", "NIGHTS"}:
        query_system_prompts_cl[k] = v + "\n\nFinally, represent the target image as image embeddings."
    elif k in {"WebQA", "EDIS", "OVEN"}:
        query_system_prompts_cl[k] = v + "\n\nFinally, represent the target as image-text multimodal embeddings."
    elif k in {"Wiki-SS-NQ"}:
        query_system_prompts_cl[k] = v + "\n\nFinally, represent the target as document embeddings."
    else:
        print(f"Task {k} is not categorized correctly or missing in query_system_prompts_base.")


query_user_prompts_clonly = {
    "CIRR": """Modification instruction: {query}

Represent the target image based on the base image and the modification instruction as image embeddings.""",
    "FashionIQ": """Modification instruction: {query}

Represent the target image based on the base image and the modification instruction as image embeddings.""",

    "VisDial": """Dialogue: {query}

Represent the target image based on the dialogue as image embeddings.""",
    "VisualNews_i2t": """Generate text embeddings for the target COCO-style caption of the image.""",
    "WebQA": """Question: {query}

Generate image-text multimodal embeddings for the target retrieval text-image pair which can answer the question.""",
    "EDIS": """News caption: {query}

Generate image-text multimodal embeddings for the target retrieval text-image based on the provided news caption.""",
    "Wiki-SS-NQ": """Question: {query}

Generate a description of the target retrieval document screenshot based on the provided question.""",
    "OVEN": """Question: {query}

Generate image-text multimodal embeddings for the target retrieval text-image pair which can answer the given visual question.""",
    "VisualNews_t2i": """News text: {query}

Generate image embeddings for the target news image based on the provided news text.""",
    "MSCOCO_t2i": """Caption: {query}

Generate image embeddings for the target image based on the provided COCO-style caption.""",
    "MSCOCO_i2t": """Generate text embeddings for the target COCO-style caption of the image.""",

    "NIGHTS": """Generate image embeddings for the given image.""",

    # grounding
    "Visual7W-Pointing": """Based on the provided image, identify the object or region the below question refers to, and generate image embeddings for the object or region.

Question: {query}""",
    "RefCOCO": """Based on the provided image, identify the object or region the below expression refers to, and generate image embeddings for the object or region.

Expression: {query}""",
    "RefCOCO-Matching": """Based on the provided image, identify the object or region the below expression refers to, and generate image embeddings for the object or region.

Expression: {query}""",
}

query_system_prompts_clonly = {

# Retrieval tasks

    "VisDial": """You are a Vision Language Model specialized in representing images based on dialogues.
Your task is to analyze the provided dialogue about an image and generate image embeddings for the target image based on the dialogue.""",
    "VisualNews_t2i": """You are a Vision Language Model specialized in representing news images based on provided news text.
Your task is to analyze the provided news text, and generate image embeddings for the target news image based on the text.""",
    "MSCOCO_i2t": """You are a Vision Language Model specialized in COCO-style image-to-text retrieval.
Your task is to analyze the provided image and generate text embeddings for the target COCO-style caption of the image.""",
    "MSCOCO_t2i": """You are a Vision Language Model specialized in COCO-style text-to-image retrieval.
Your task is to analyze the provided COCO-style caption and generate image embeddings for the target image based on the caption.""",
    "WebQA": """You are a Vision Language Model specialized in retrieving relevant text-image pairs to answer the provided question.
Your task is to analyze the provided question and generate text-image multimodal embeddings for the target retrieval text-image pair which can answer the question.""",
    "EDIS": """You are a Vision Language Model specialized in retrieving relevant text-image pairs given the provided news caption.
Your task is to analyze the provided news caption and generate image-text multimodal embeddings for the target retrieval text-image based on the provided news caption.""",
    "Wiki-SS-NQ": """You are a Vision Language Model specialized in document screenshots retrieval.
Your task is to analyze the provided question and generate document embeddings for the target retrieval document screenshot.""",
    "VisualNews_i2t": """You are a Vision Language Model specialized in news image-to-text retrieval.
Your task is to analyze the provided news image and generate text embeddings for the target COCO-style caption of the image.""",
    "CIRR": """You are a Vision Language Model specialized in composed image retrieval.
Your task is to analyze the provided base image and a modification instruction, which describes the changes to be applied to the base image in order to get a target image, and generate image embeddings for the target image based on the base image and the modification instruction.""",
    "FashionIQ": """You are a Vision Language Model specialized in composed image retrieval in fashion domain.
Your task is to analyze the provided base fashion image and a modification instruction which describes the changes to be applied to the base image in order to get a target fashion image, and generate image embeddings for the target fashion image based on the base image and the modification instruction.""",
    "NIGHTS": """You are a Vision Language Model specialized in embedding images.
Your task is to analyze the provided base image and generate image embeddings for the image.""",
    "OVEN": """You are a Vision Language Model specialized in text-image retrieval for visual question answering.
Your task is to analyze the provided image and question, and generate image-text multimodal embeddings for the target retrieval text-image pair which can answer the given visual question.""",

# Grounding tasks
    "MSCOCO": """You are a Vision Language Model specialized in object grounding.
Your task is to analyze the provided query and locate the object or region that the query refers to in the image, and generate image embeddings for the object or region.""",
    "Visual7W-Pointing": """You are a Vision Language Model specialized in object grounding.
Your task is to analyze the provided query and locate the object or region that the query refers to in the image, and generate image embeddings for the object or region.""",
    "RefCOCO": """You are a Vision Language Model specialized in object grounding.
Your task is to analyze the provided query and locate the object or region that the query refers to in the image, and generate image embeddings for the object or region.""",
    "RefCOCO-Matching": """You are a Vision Language Model specialized in object grounding.
Your task is to analyze the provided query and locate the object or region that the query refers to in the image, and generate image embeddings for the object or region.""",
}

query_user_prompts_cot = {}

for task in tasks:
    if task in task_categories['vqa']:
        query_user_prompts_cot[task] = """Given the image and the below question, answer the question based on the image. Let's think step by step.

Question: {query}"""
    elif task in {"ImageNet-1K", "ImageNet_1K", "ImageNet-A", "ImageNet-R", "VOC2007", "ObjectNet"}:
        query_user_prompts_cot[task] = """Given an image, identify the **main object category** it belongs to. Let's think step by step."""
    elif task in {"Country211"}:
        query_user_prompts_cot[task] = """Given an image, identify the **country** where it was taken. Let's think step by step."""
    elif task in {"HatefulMemes"}:
        query_user_prompts_cot[task] = """Given an image, determine if it contains hateful speech. Identify the texts in the image and Let's think step by step."""
    elif task in {"SUN397", "Place365"}:
        query_user_prompts_cot[task] = """Given an image, identify the **scene** it depicts. Let's think step by step."""
    elif task in {"N24News"}:
        query_user_prompts_cot[task] = """Given an image and its associated news text, identify the **main domain** of the news. Let's think step by step.

News text: {query}"""
### TODO: for retrieval and grounding, currently they only have descriptions. have to add cot later
    elif task in {"VisDial"}:
        query_user_prompts_cot[task] = """Given the dialogue about an image, generate a description of the image. Let's think step by step.

Dialogue: {query}"""
    elif task in {"CIRR"}:
        query_user_prompts_cot[task] = """Given a base image and a modification instruction of how to modify the base image to get a target image, generate a description of the target image. Let's think step by step.

Modification instruction: {query}"""

    elif task in {"FashionIQ"}:
        query_user_prompts_cot[task] = """Given a base fashion image and a modification instruction of how to modify the garment in the base fashion image to get the target garment, generate a description of the target garment. Let's think step by step.

Modification instruction: {query}"""

    elif task in {"VisualNews_t2i"}:
        query_user_prompts_cot[task] = """Given the news text, expand it into a more detailed and concise description of the target news image. Let's think step by step.

News text: {query}"""

    elif task in {"VisualNews_i2t"}:
        query_user_prompts_cot[task] = """Given a news image, generate a concise and informative description of the news image. Let's think step by step."""

    elif task in {"MSCOCO_t2i"}:
        query_user_prompts_cot[task] = """Given a COCO-style caption, generate a more detailed and concise description of the target image based on the caption. Let's think step by step.

Caption: {query}"""
    elif task in {"MSCOCO_i2t"}:
        query_user_prompts_cot[task] = """Given an image, first generate a detailed and informative description of the image and analyze what object/activity/visual features should be included in the caption, and finally generate a COCO-style caption based on your reasoning. Let's think step by step."""
    elif task in {"MSCOCO", "Visual7W-Pointing", "RefCOCO", "RefCOCO-Matching"}:
        query_user_prompts_cot[task] = """Given an image and a query, identify the object or region in the image that the query refers to, and generate a concise description of the object or region. Let's think step by step.

Query: {query}"""
    elif task in {"WebQA", "OVEN"}:
        query_user_prompts_cot[task] = """Given a question, determine what kind of image-text pair would help answer the question. Let's think step by step.

Question: {query}"""
    
    elif task in {"EDIS"}:
        query_user_prompts_cot[task] = """Given a news text, expand the caption with more visual and textual cues that would help retrieve a matching image-text pair. Let's think step by step.

News text: {query}"""
    elif task in {"Wiki-SS-NQ"}:
        query_user_prompts_cot[task] = """Given a question, explain what information is needed to answer the question. Let's think step by step.

Question: {query}"""
    elif task in {"NIGHTS"}:
        query_user_prompts_cot[task] = """Given a query image, generate a concise and informative description of what the target image may look like. Let's think step by step."""


target_user_prompts_cot = {}

for task in tasks:
    if task in {"MSCOCO_i2t"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["MSCOCO_t2i"]
    elif task in {"MSCOCO_t2i"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["MSCOCO_i2t"]
    elif task in {"VisualNews_i2t"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["VisualNews_t2i"]
    elif task in {"VisualNews_t2i"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["VisualNews_i2t"]
    elif task in {"CIRR"}:
        target_user_prompts_cot[task] = query_user_prompts_cot['MSCOCO_i2t']
    elif task in {"FashionIQ"}:
        target_user_prompts_cot[task] = """Given a fashion image, first generate a detailed description of the garment in the image, and then generate a summarization on the description. Focus solely on the garment and ignore the background or the person in the image. Let's think step by step."""
    elif task in {"VisDial"}:
        target_user_prompts_cot[task] = query_user_prompts_cot['MSCOCO_i2t']
    elif task in {"NIGHTS"}:
        target_user_prompts_cot[task] = query_user_prompts_cot['NIGHTS']
    elif task in {"OVEN", "WebQA"}:
        target_user_prompts_cot[task] = """Given an wiki image and a short text description, first describe the image in detail incorporating relevant information from the accompanying text, then produce a short summary.
        
Text: {query}"""
    elif task in {"Wiki-SS-NQ"}:
        target_user_prompts_cot[task] = """Given a document screenshot, first describe the content of the document in detail, then produce a short summary."""
    elif task in {"EDIS"}:
        target_user_prompts_cot[task] = """Given a news image and a text description, first describe the news image in detail incorporating relevant information from the accompanying text, then produce a short summary.
        
News text: {query}"""
    elif task in {"MSCOCO", "Visual7W-Pointing", "RefCOCO"}:
        target_user_prompts_cot[task] = """Given an image, first generate a detailed and informative description of the image, and then generate a summarization based on the description. Let's think step by step."""
    elif task in {"RefCOCO-Matching"}:
        target_user_prompts_cot[task] = query_user_prompts_cot["MSCOCO"]
    elif task in task_categories['vqa'] | task_categories['classification']:
        target_user_prompts_cot[task] = query_user_prompts_cot[task]    

    # else:
        # query_user_prompts_cot[task] = query_user_prompts_base[task]

# def format_query(task_name, query=None):
#     if task_name in task_categories['vqa']:
#         template = "Question: {query}"
#     elif task_name in {"VOC2007", "ImageNet_1K", "ImageNet-A", "ImageNet-R", "ObjectNet"}:
#         template = "Question: What is the main object category in the image?"
#     elif task_name in {"Country211"}:
#         template = "Question: What is the country where the image was taken?"
#     elif task_name in {"HatefulMemes"}:
#         template = "Question: does the image contain hateful speech? Answer with 'Yes' or 'No'."
#     elif task_name in {"SUN397", "Place365"}:
#         template = "Question: What is the scene in the provided image?"
#     elif task_name in {"N24News"}:
#         template = """Question: What is the main domain of the news image and its text below?
        
# News text: {query} 

# Answer with a single word or short phrase."""
#     elif task_name in {"VisDial"}:
#         template = "Dialogue: {query}"
#     elif task_name in {"CIRR", "FashionIQ"}:
#         template = """Modification instruction: {query}"""
#     elif task_name in {"VisualNews_t2i"}:
#         template = """News text: {query}"""
#     elif task_name in


tasks = ['ImageNet-1K', 'N24News', 'HatefulMemes', 'VOC2007', 'SUN397', 'A-OKVQA', 'MSCOCO', 
         'Place365', 'ImageNet-A', 'ImageNet-R', 'ObjectNet', 'Country211', 'OK-VQA', 'RefCOCO', 
         'DocVQA', 'InfographicsVQA', 'ChartQA', 'NIGHTS', 'FashionIQ', 'ScienceQA', 'Visual7W', 
         'VizWiz', 'GQA', 'TextVQA', 'VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 
         'MSCOCO_i2t', 'Wiki-SS-NQ', 'WebQA', 'OVEN', 'EDIS', 'RefCOCO-Matching', 'Visual7W-Pointing']


target_system_prompts_base = {
# VQA
    "OK-VQA": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "A-OKVQA": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "DocVQA": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "InfographicsVQA": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "ScienceQA": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "Visual7W": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "VizWiz": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "GQA": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "TextVQA": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "ChartQA": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",

# Classification
    "ImageNet_1K": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "ImageNet-1K": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "ImageNet-A": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "ImageNet-R": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "N24News": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given domain name.""",
    "Place365": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given scene name.""",
    "ObjectNet": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "Country211": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given country name.""",
    "VOC2007": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "HatefulMemes": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given text.""",
    "SUN397": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given scene name.""",


# Retrieval
    "CIRR": """You are a Vision Language Model specialized in providing image embeddings.
Your task is to generate image embeddings for the given image.""",
    "FashionIQ": """You are a Vision Language Model specialized in providing image embeddings.
Your task is to generate image embeddings for the given fashion image.""",
    "VisDial": """You are a Vision Language Model specialized in providing image embeddings.
Your task is to generate image embeddings for the given image.""",
    "MSCOCO_i2t": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given image caption.""",
    "NIGHTS": """You are a Vision Language Model specialized in providing image embeddings.
Your task is to generate image embeddings for the given image.""",
    "OVEN": """You are a Vision Language Model specialized in providing image-text multimodal embeddings.
Your task is to generate image-text multimodal embeddings for the given image and text.""",
    "VisualNews_t2i": """You are a Vision Language Model specialized in providing image embeddings.
Your task is to generate image embeddings for the given image.""",
    "VisualNews_i2t": """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the given image caption.""",
    "WebQA": """You are a Vision Language Model specialized in providing image-text multimodal embeddings.
Your task is to generate image-text multimodal embeddings for the given image and text.""",
    "EDIS": """You are a Vision Language Model specialized in providing image-text multimodal embeddings.
Your task is to generate image-text multimodal embeddings for the given image and text.""",
    "Wiki-SS-NQ": """You are a Vision Language Model specialized in providing document embeddings.
Your task is to generate document embeddings for the given document screenshot.""",
    "MSCOCO_t2i": """You are a Vision Language Model specialized in providing image embeddings.
Your task is to generate image embeddings for the given image.""",

# Grounding
    "MSCOCO": """You are a Vision Language Model specialized in providing image embeddings.
Your task is to generate image embeddings for the given image.""",
    "RefCOCO": """You are a Vision Language Model specialized in providing image embeddings.
Your task is to generate image embeddings for the given image.""",
    "RefCOCO-Matching": """You are a Vision Language Model specialized in providing image embeddings.
Your task is to generate image embeddings for the object or region in the image that the given expression refers to.""",
    "Visual7W-Pointing": """You are a Vision Language Model specialized in providing image embeddings.
Your task is to generate image embeddings for the given image.""",
}

target_system_prompts_with_question = {}

for k, v in target_system_prompts_base.items():
    if k in task_categories['vqa'] or k in task_categories['classification']:
        target_system_prompts_with_question[k] = """You are a Vision Language Model specialized in providing text embeddings.
Your task is to generate text embeddings for the provided question and answer."""
    else:
        target_system_prompts_with_question[k] = v


target_user_prompts = {}

for task in tasks:
    if task in task_categories['vqa']:
        target_user_prompts[task] = """Represent the following question-answer pair as text embeddings.
Question: {query}
Answer: {answer}."""
    elif task in {"VisualNews_i2t"}:
        target_user_prompts[task] = """Represent the following news image caption as text embeddings.
Caption: {query}."""
    elif task in {"MSCOCO_i2t"}:
        target_user_prompts[task] = """Represent the following COCO-style caption as text embeddings.
Caption: {query}."""
    elif task in {"ImageNet_1K", "ImageNet-1K", "ImageNet-A", "ImageNet-R", "ObjectNet", "VOC2007"}:
        target_user_prompts[task] = """Represent the following object category as text embeddings.
Object category: {query}."""
    elif task in {"Country211"}:
        target_user_prompts[task] = """Represent the following country name as text embeddings.
Country name: {query}."""
    elif task in {"HatefulMemes"}:
        target_user_prompts[task] = """Represent the following question-answer pair as text embeddings.
Question: Does the image contain hateful speech?
Answer: {query}."""
    elif task in {"SUN397", "Place365"}:
        target_user_prompts[task] = """Represent the following scene name as text embeddings.
Scene name: {query}."""
    elif task in {"N24News"}:
        target_user_prompts[task] = """Represent the following domain name as text embeddings.
Domain name: {query}."""
    elif task in {"VisDial", "MSCOCO", "VisualNews_t2i", "CIRR", "NIGHTS", "MSCOCO_t2i", "Visual7W-Pointing", "RefCOCO"}:
        target_user_prompts[task] = """Represent the given image as image embeddings."""
    elif task in {"FashionIQ"}:
        target_user_prompts[task] = """Represent the given fashion image as image embeddings."""
    elif task in {"WebQA", "EDIS", "OVEN"}:
        target_user_prompts[task] = """Represent the given image and text as image-text multimodal embeddings.
Text: {query}."""
    elif task in {"Wiki-SS-NQ"}:
        target_user_prompts[task] = """Represent the given document screenshot as document embeddings."""
    elif task in {"RefCOCO-Matching"}:
        target_user_prompts[task] = """Represent the object or region in the image that the given expression refers to as image embeddings.
Expression: {query}."""





def format_target_user_prompt(task, query=None, answer=None, with_question=False, with_description=False):
    if task in task_categories['vqa']:
        if with_question:
            output = f"""Represent the following question-answer pair as text embeddings.
Question: {query}
Answer: {answer}."""
        else:
            output = f"""Represent the following text answer as text embeddings.
Answer: {answer}."""
    elif task in {"VisualNews_i2t"}:
        if with_description:
            output = query_user_prompts_cot["VisualNews_t2i"].format(query=answer)
        else:
            output = f"""Represent the following news image caption as text embeddings.
Caption: {answer}."""
    elif task in {"MSCOCO_i2t"}:
        if with_description:
            output = query_user_prompts_cot["MSCOCO_t2i"].format(query=answer)
        else:
            output = f"""Represent the following COCO-style caption as text embeddings.
Caption: {answer}."""
    elif task in {"ImageNet_1K", "ImageNet-1K", "ImageNet-A", "ImageNet-R", "ObjectNet", "VOC2007"}:
        output = f"""Represent the following object category as text embeddings.
Object category: {answer}."""
    elif task in {"Country211"}:
        output = f"""Represent the following country name as text embeddings.
Country name: {answer}."""
    elif task in {"HatefulMemes"}:
        output = f"""Represent the following question-answer pair as text embeddings.
Question: Does the image contain hateful speech?
Answer: {answer}."""
    elif task in {"SUN397", "Place365"}:
        output = f"""Represent the following scene name as text embeddings.
Scene name: {answer}."""
    elif task in {"N24News"}:
        output = f"""Represent the following domain name as text embeddings.
Domain name: {answer}."""
    # elif task in {"VisDial", "MSCOCO", "VisualNews_t2i", "CIRR", "NIGHTS", "MSCOCO_t2i", "Visual7W-Pointing", "RefCOCO"}:
    #     output = f"""Represent the given image as image embeddings."""
    elif task in {"VisDial"}:
        if with_description:
            output = query_user_prompts_cot["MSCOCO_i2t"]
        else:
            output = f"""Represent the given image as image embeddings."""
    elif task in {"CIRR"}:
        if with_description:
            output = query_user_prompts_cot["MSCOCO_i2t"]
        else:
            output = f"""Represent the given image as image embeddings."""
    elif task in {"VisualNews_t2i"}:
        if with_description:
            output = query_user_prompts_cot["VisualNews_i2t"]
        else:
            output = f"""Represent the given image as image embeddings."""
    elif task in {"MSCOCO_t2i"}:
        if with_description:
            output = query_user_prompts_cot["MSCOCO_i2t"]
        else:
            output = f"""Represent the given image as image embeddings."""
    elif task in {"NIGHTS"}:
        if with_description:
            output = query_user_prompts_cot["NIGHTS"]
        else:
            output = f"""Represent the given image as image embeddings."""
    elif task in {"OVEN", "WebQA"}:
        if with_description:
            output = query_user_prompts_cot[task].format(query=answer)
        else:
            output = f"""Represent the given image and text as image-text multimodal embeddings.

Text: {answer}."""
    elif task in {"FashionIQ"}:
        if with_description:
            output = """Given a fashion image, first generate a detailed description of the garment in the image, and then generate a COCO-style caption based on the description. Focus solely on the garment and ignore the background or the person in the image. Let's think step by step."""
        else:
            output = f"""Represent the given fashion image as image embeddings."""
    elif task in {"EDIS"}:
        if with_description:
            output = f"""Given a news image and a text description, first describe the news image in detail incorporating relevant information from the accompanying text, then produce a short summary.

News text: {answer}"""
        else:
            output = f"""Represent the given image and text as image-text multimodal embeddings.

Text: {answer}."""
    elif task in {"Wiki-SS-NQ"}:
        if with_description:
            output = """Given a document screenshot, first describe the content of the document in detail, then produce a short summary."""
        else:
            output = f"""Represent the given document screenshot as document embeddings."""
    elif task in {"RefCOCO-Matching"}:
        if with_description:
            output = query_user_prompts_cot[task].format(query=answer)
        else:
            output = f"""Represent the object or region in the image that the given expression refers to as image embeddings.

Expression: {answer}."""
    elif task in {"MSCOCO", "Visual7W-Pointing", "RefCOCO"}:
        if with_description:
            output = query_user_prompts_cot["MSCOCO_i2t"]
        else:
            output = f"""Represent the given image as image embeddings."""
    else:
        raise ValueError(f"Task {task} is not recognized or not supported for target user prompts.")


    return output
        



if __name__ == "__main__":
    for task in tasks:
        print(task)
        if task not in query_user_prompts_cot:
            print(f"Task {task} is missing in query_user_prompts_cot.")
        print(format_target_user_prompt(task, query="", answer="", with_question=False, with_description=True))
        # if task not in query_system_prompts_cl:
        #     print(f"Task {task} is missing in query_system_prompts_cl.")
        # if task not in query_user_prompts_base:
        #     print(f"Task {task} is missing in query_user_base.")
        # if task not in query_user_prompts_cl:
        #     print(f"Task {task} is missing in query_user_cl.")
        # if task not in target_system_prompts_base:
        #     print(f"Task {task} is missing in target_system_prompts_base.")
        # if task not in target_system_prompts_with_question:
        #     print(f"Task {task} is missing in target_system_prompts_with_question.")
        # if task not in target_user_prompts:
        #     print(f"Task {task} is missing in target_user_prompts.")
        # if task not in query_system_prompts_clonly:
        #     print(f"Task {task} is missing in query_system_prompts_clonly.")
        # if task not in query_system_prompts_base:
        #     print(f"Task {task} is missing in query_system_prompts_base.")
        # if task not in query_system_prompts_cl:
        #     print(f"Task {task} is missing in query_system_prompts_cl.")

    print("Finished sanity check for tasks and prompts.")