VIDORE_QA_RETRIEVAL_DATASETS = [
    "ViDoRe_arxivqa",
    "ViDoRe_docvqa",
    "ViDoRe_infovqa",
    "ViDoRe_tabfquad",
    "ViDoRe_tatdqa",
    "ViDoRe_shiftproject",
    "ViDoRe_syntheticDocQA_artificial_intelligence",
    "ViDoRe_syntheticDocQA_energy",
    "ViDoRe_syntheticDocQA_government_reports",
    "ViDoRe_syntheticDocQA_healthcare_industry",
    "ViDoRe_esg_reports_human_labeled_v2",
    "ViDoRe_biomedical_lectures_v2_multilingual",
    "ViDoRe_economics_reports_v2_multilingual",
    "ViDoRe_esg_reports_v2_multilingual",

    "ViDoSeek-page",
    "ViDoSeek-doc",
    "MMLongBench-doc",
    "MMLongBench-page"
]

VISRAG_QA_RETRIEVAL_DATASETS = [
    "VisRAG_ArxivQA",
    "VisRAG_ChartQA",
    "VisRAG_MP-DocVQA",
    "VisRAG_SlideVQA",
    "VisRAG_InfoVQA",
    "VisRAG_PlotQA"
]

prompts = {
    "colpali_train_set": 
"""
You are given a visual document and a question about it. Your task is to generate an explanation and the final answer in a specific format.

Instructions:

- Begin with a detailed description of the visual document, focusing on text, layout, and key elements relevant to answering the question.
- Then, reason step-by-step to find the answer.
- Wrap the description and reasoning inside <think> ... </think> tags. Keep them succinct (2–3 sentences).
- On a new line, output the final answer in this exact format: Answer: <answer>
- You are provided with the ground-truth answer for reference. Use it to verify your reasoning but do not mention it explicitly in your explanation.
- Rephrase the final answer so that it preserves the exact meaning of the original but may differ in wording or phrasing. Do not add, remove, or alter factual content.

---

EXAMPLE:

Question: What is the duration of the course mentioned in the image?  
Ground-truth answer: five to ten hours, not including field trips  

<think>
The document is a typed instructional text titled "HOW TO USE THESE MATERIALS." It describes a course on energy conservation for practicing contractors, prepared by members of the NC Community College staff familiar with home construction. The course duration is stated in the opening paragraph, followed by objectives on topics such as energy-efficient home design, passive solar techniques, insulation, heat loss prevention, heat pumps, and other energy-saving devices.  
From the first paragraph, it clearly says the course will take "from five to ten hours, not including field trips." Since the question is about duration, that phrase directly answers it.
</think>  
Answer: between five and ten hours, excluding field trips

---

Now, answer for the following visual document and question:

Question: {query}
Ground-truth answer: {target}
Visual document: <image>""",
    "visrag-indomain":
"""You are given a document image. Your task is to generate a concise factual description based only on the visible content.

Instructions:

- Extract facts only from the document — do not make assumptions or add external information.
- Write the description in 3–4 sentences
- Avoid subjective opinions, speculation, or stylistic commentary.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output a summary in this exact format: Answer: <answer>

---

EXAMPLE:

<think>The document presents information on the "Most Spoken Languages in the World". It lists languages by the number of native speakers, with Chinese having 935 million native speakers, followed by Spanish with 387 million, and English with 365 million. The document also shows the percentage of the world population for each language and indicates where each language is mainly spoken around the world.</think>
Summary: The document lists the most spoken languages in the world by native speakers, including Chinese, Spanish, and English, and shows their percentage of the world population and main regions where they are spoken.

---

Now, answer for the following visual document:
Visual document: <image>""",

    "video_caption_300k":
"""You are given a video. Your task is to generate a thorough and detailed description that describes the video content.

Instructions:
- Focus on the visual elements, actions, and interactions occurring in the video.
- Focus on the temporal sequence of events.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Wrap the description inside <think> ... </think> tags.
- On a new line, output a summary in this exact format: Summary: <summary>

---

EXAMPLE:

<think>The video features a medium shot of an older woman with short white hair and fair skin, wearing a white shirt and a purple floral jacket. She speaks directly to the camera, occasionally looking to the right. The setting includes a stone fireplace with a black grate and tools, and a doorway in the background. A small black microphone is clipped to her jacket. The handheld camera gradually zooms in on her as she talks.</think>
Summary: The video shows an older woman speaking to the camera in front of a fireplace as the camera zooms in.

---

Now, generate a detailed description for the following video:

Video: <video>""",

    "video_qa_240k":
"""You are given a video and a question about it. Your task is to first generate a detailed description of the video content and the reasoning process to answer the question, and finally provide the answer in a specific format.

Instructions:
- Begin with a thorough and concise description of the video, focusing on visual elements, actions, and temporal sequence.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Then, reason step-by-step to find the answer to the question.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output the final answer with a brief explanation in this exact format: Answer: <answer>

---

EXAMPLE:

Question: What is the role of the man in the suit in the conversation?

<think>The video shows a Fox News segment with a group of fencers and three men at the center of the frame. The man in the suit is holding a microphone with a Fox News logo and actively engaging with the other two men, who are dressed as a fencer and a coach. This visual evidence, combined with his central position and professional attire, strongly indicates that he is the host or anchor of the program. He is the one facilitating the discussion, while the others are the guests being interviewed about the news topic of a university's fencing ban, which is visible on the news ticker.</think>
Answer: The man in the suit appears to be hosting the segment as he holds papers and gestures while speaking.

---

Now, answer for the following video and question:

Question: {query}
Video: <video>""",

    "YouCook2":
"""You are given a cooking video. Your task is to generate a thorough and detailed description of the video content.

Instructions:
- Focus on the cooking actions, ingredients, and interactions visible in the video.
- Describe the temporal sequence of steps clearly, capturing how the dish is prepared from start to finish.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Wrap the detailed description inside <think> ... </think> tags.
- On a new line, output a concise summary of the video in this exact format:
Summary: <summary>


EXAMPLE:

<think>The video begins with a close-up of a wooden cutting board where a chef slices red bell peppers into thin strips. Next, the camera shifts to a stovetop view where olive oil is poured into a heated pan, followed by the addition of minced garlic. The peppers are added and sautéed until softened. The chef then sprinkles salt and pepper while stirring with a wooden spatula. In the final scene, the sautéed peppers are transferred into a serving bowl, garnished with fresh parsley, and placed on a countertop alongside toasted bread slices.</think>
Summary: The video shows a chef slicing bell peppers and sautéing them with garlic and olive oil, finishing with a parsley garnish and serving them alongside bread.


Now, generate a detailed description for the following video:

Video: <video>""",}



for dataset in VIDORE_QA_RETRIEVAL_DATASETS + VISRAG_QA_RETRIEVAL_DATASETS:
    prompts[dataset] = prompts["visrag-indomain"]

for dataset in ["MSVD", "MSR-VTT", "DiDeMo", "VATEX"]:
    prompts[dataset] = prompts["video_caption_300k"]