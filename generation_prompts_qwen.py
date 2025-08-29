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

<think>The document is a typed instructional text titled "HOW TO USE THESE MATERIALS." It describes a course on energy conservation for practicing contractors, prepared by members of the NC Community College staff familiar with home construction. The course duration is stated in the opening paragraph, followed by objectives on topics such as energy-efficient home design, passive solar techniques, insulation, heat loss prevention, heat pumps, and other energy-saving devices.  
From the first paragraph, it clearly says the course will take "from five to ten hours, not including field trips." Since the question is about duration, that phrase directly answers it.</think>
Answer: between five and ten hours, excluding field trips

---

Now, answer for the given visual document and the following question:

Question: {query}
Ground-truth answer: {target}""",
    "visrag-indomain":
"""You are given a document image. Your task is to generate a concise factual description based only on the visible content.

Instructions:

- Extract facts only from the document — do not make assumptions or add external information.
- Write the description in 3–4 sentences
- Avoid subjective opinions, speculation, or stylistic commentary.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output a one-sentence summary in this exact format: Answer: <answer>

---

EXAMPLE:

<think>The document presents information on the "Most Spoken Languages in the World". It lists languages by the number of native speakers, with Chinese having 935 million native speakers, followed by Spanish with 387 million, and English with 365 million. The document also shows the percentage of the world population for each language and indicates where each language is mainly spoken around the world.</think>
Summary: The document lists the most spoken languages in the world by native speakers, including Chinese, Spanish, and English, and shows their percentage of the world population and main regions where they are spoken.

---

Now, answer for the given visual document""",

    "video_caption_300k":
"""You are given a video. Your task is to generate a thorough and detailed description that describes the video content.

Instructions:
- Focus on the visual elements, actions, and interactions occurring in the video.
- Focus on the temporal sequence of events.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Wrap the description inside <think> ... </think> tags.
- On a new line, output a one-sentence summary in this exact format: Summary: <summary>

---

EXAMPLE:

<think>The video features a medium shot of an older woman with short white hair and fair skin, wearing a white shirt and a purple floral jacket. She speaks directly to the camera, occasionally looking to the right. The setting includes a stone fireplace with a black grate and tools, and a doorway in the background. A small black microphone is clipped to her jacket. The handheld camera gradually zooms in on her as she talks.</think>
Summary: The video shows an older woman speaking to the camera in front of a fireplace as the camera zooms in.

---

Now, generate a detailed description for the given video.""",

    "video_qa_240k":
"""You are given a video and a question about it. Your task is to first generate a detailed description of the video content and the reasoning process to answer the question, and finally provide the answer in a specific format.

Instructions:
- Begin with a thorough and concise description of the video, focusing on visual elements, actions, and temporal sequence.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Then, reason step-by-step to find the answer to the question.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output the final answer with a brief one-sentence explanation in this exact format: Answer: <answer>

---

EXAMPLE:

Question: What is the role of the man in the suit in the conversation?

<think>The video shows a Fox News segment with a group of fencers and three men at the center of the frame. The man in the suit is holding a microphone with a Fox News logo and actively engaging with the other two men, who are dressed as a fencer and a coach. This visual evidence, combined with his central position and professional attire, strongly indicates that he is the host or anchor of the program. He is the one facilitating the discussion, while the others are the guests being interviewed about the news topic of a university's fencing ban, which is visible on the news ticker.</think>
Answer: The man in the suit appears to be hosting the segment as he holds papers and gestures while speaking.

---

Now, answer for the given video with the following question:

Question: {query}""",

    "YouCook2":
"""You are given a cooking video. Your task is to generate a thorough and detailed description of the video content.

Instructions:
- Focus on the cooking actions, ingredients, and interactions visible in the video.
- Describe the temporal sequence of steps clearly, capturing how the dish is prepared from start to finish.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Wrap the detailed description inside <think> ... </think> tags.
- On a new line, output a concise one-sentence summary of the video in this exact format:
Summary: <summary>


EXAMPLE:

<think>The video begins with a close-up of a wooden cutting board where a chef slices red bell peppers into thin strips. Next, the camera shifts to a stovetop view where olive oil is poured into a heated pan, followed by the addition of minced garlic. The peppers are added and sautéed until softened. The chef then sprinkles salt and pepper while stirring with a wooden spatula. In the final scene, the sautéed peppers are transferred into a serving bowl, garnished with fresh parsley, and placed on a countertop alongside toasted bread slices.</think>
Summary: The video shows a chef slicing bell peppers and sautéing them with garlic and olive oil, finishing with a parsley garnish and serving them alongside bread.


Now, generate a detailed description for the given video.""",

"MVBench": """You are given a video and a multiple-choice question about it. Your task is to first generate a detailed description of the video content and your reasoning process to determine the correct option, and finally provide the answer in a specific format.

Instructions:
- Begin with a thorough and concise description of the video, focusing on visual elements, actions, and temporal sequence.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Then, reason step-by-step to evaluate each option and determine which one best fits the video content.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output the final answer chosen from the provided options in this exact format: Answer: <option>

---

EXAMPLE:

Question: Why did Castle dress like a fairy when he was speaking to Emily?  
Options:
(A) To get her to trust him
(B) He secretly loved fairies
(C) He lost a bet with Emily

<think>The video shows Castle walking into a classroom wearing a pink fairy costume with wings and a wand. Emily looks upset and hesitant to talk to him at first. Castle kneels down to her level, speaks gently, and smiles reassuringly. After a moment, Emily starts talking and visibly relaxes. None of the other children are dressed similarly, ruling out option D. There’s no mention or visual cue of a bet, a school event, or Mrs. Ruiz forcing him. The interaction and Castle’s behavior strongly suggest he dressed up intentionally to make Emily comfortable and earn her trust.</think>
Answer: A. To get her to trust him

---

Now, answer for the given video with the following question:

Question: {query}""",

"Video-MME": """You are given a video and a multiple-choice question about it. Your task is to first generate a detailed description of the video content and your reasoning process to determine the correct option, and finally provide the answer in the specified format.

Instructions:
- Begin with a thorough and concise description of the video, focusing on visual elements, actions, and temporal sequence.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Then, reason step-by-step to evaluate each option and identify the correct one.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output the final answer in this exact format:  
  `Answer: <correct option text>`

---

EXAMPLE:

Question: When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?  
Options:
A. Apples.
B. Candles.
C. Berries.
D. The three kinds are of the same number.

<think>The video shows a close-up of a Christmas tree being decorated in a traditional German style. In the initial scene, the tree has numerous red apples placed across several branches, a few candles positioned evenly, and only a handful of red berries. By visually comparing the quantities, apples clearly outnumber the candles and berries.</think>
Answer: Apples

---

Now, answer for the given video with the following question:

Question: {query}""",

"NExTQA": """You are given a video and a multiple-choice question about it. Your task is to first generate a detailed description of the video content and your reasoning process to determine the correct option, and finally provide the answer in the specified format.

Instructions:
- Begin with a thorough and concise description of the video, focusing on visual elements, actions, and temporal sequence.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Then, reason step-by-step to evaluate each option and identify the correct one.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output the final answer in this exact format:  
Answer: <correct option>

---

EXAMPLE:

Question: What did the baby do after throwing the green cup away while on the floor near the end?  
Options:  
(A) clap proudly
(B) the lady sitting down
(C) lay on floor
(D) just picked it up
(E) crawl

<think>The video shows a baby sitting on the floor, playing with a green cup. Near the end, the baby tosses the cup aside. Immediately afterward, the baby claps its hands enthusiastically, smiling and looking toward the adult in the room. The baby does not lie on the floor, crawl, or retrieve the cup, and the lady remains seated without interacting. This confirms that the baby clapped proudly after throwing the cup.</think>
Answer: (A) clap proudly

Now, answer for the given video with the following question:

Question: {query}""",

"ActivityNetQA": """You are given a video and a yes/no question about it. Your task is to first generate a detailed description of the video content and your reasoning process to determine the correct answer, and finally provide the answer in the specified format.

Instructions:
- Begin with a thorough and concise description of the video, focusing on visual elements, actions, and temporal sequence.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Then, reason step-by-step to verify the correct yes/no answer.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output the final answer in this exact format:  
  `Answer: Yes` or `Answer: No`

---

EXAMPLE

Question: Does the girl in black clothes have long hair?\nOptions:\nyes\nno

<think>The video shows a girl wearing a black T-shirt and black pants standing near a table. Her hair clearly extends past her shoulders and flows down her back when she turns to the side. This confirms that she indeed has long hair.</think>
Answer: yes

---

Now, answer for the given video with the following question:

Question: {query}""",

"EgoSchema": """You are given a video and a multiple-choice question about it. Your task is to first generate a detailed description of the video content and your reasoning process to determine the correct option, and finally provide the answer in the specified format.

Instructions:
- Begin with a thorough and concise description of the video, focusing on visual elements, actions, and temporal sequence.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Then, reason step-by-step to evaluate each option and identify the correct one.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output the final answer in this exact format:  
  `Answer: <correct option text>`

---

EXAMPLE

Question: Taking into account all the actions performed by C, what can you deduce about the primary objective and focus within the video content?  
Options:
A. C is cooking.
B. C is doing laundry.
C. C is cleaning the kitchen.

<think>The video shows someone in a kitchen area. The person is chopping vegetables on a cutting board, stirring a pot on the stove, and adding spices to the mixture. There is no sign of laundry, bathroom cleaning, or dishwashing activities. The repeated focus on food preparation and cooking utensils strongly indicates that C is engaged in cooking.</think>
Answer: C is cooking

---

Now, answer for the given video with the following question:

Question: {query}""",

"HMDB51": """You are given a video showing a person performing an action. Your task is to first generate a detailed description of the video content and your reasoning process to identify the action, and finally provide the answer in the specified format.

Instructions:
- Begin with a concise description of the video, focusing on the person, their movements, and any relevant objects or context.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Then, reason step-by-step to identify the most accurate action label for the person.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output the final answer in this exact format:  
  `Answer: <action word or short phrase>`

---

**EXAMPLE**

Question: What action is the person performing in the video?  

<think>The video shows a man standing in front of a punching bag. He repeatedly throws quick punches using both hands, alternating between left and right. His stance and arm movements are consistent with practicing boxing techniques.</think>
Answer: boxing

---

Now, answer for the given video.""",

"Kinetics-700": """You are given a video, and your task is to recognize the category of the video content.

Instructions:
- Begin with a concise description of the video, focusing on the main subjects, their actions, and any relevant objects or context.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Then, reason step-by-step to determine the most accurate action or event category.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output the final answer in this exact format:
  `Answer: <category>`

---

EXAMPLE

Question: What is the category of the video content?  

<think>
The video shows a person standing at a kitchen counter with vegetables laid out. They pick up a knife and carefully slice tomatoes, onions, and peppers, placing the pieces into a bowl. The consistent cutting and preparation of food ingredients indicate that the action being performed is chopping vegetables.</think>
Answer: chopping vegetables

---

Now, answer for the given video.""",

"Breakfast": """You are given a video, and your task is to recognize the type of breakfast the person is preparing.

Instructions:
- Begin with a concise description of the video, focusing on the key ingredients, cooking actions, and tools used.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Then, reason step-by-step to determine the specific breakfast type being prepared.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output the final answer in this exact format:
  `Answer: <breakfast type>`

---

**EXAMPLE**

Question: What type of breakfast is the person preparing in the video?  

<think>
The video shows a person cracking eggs into a bowl, whisking them, and then pouring the mixture into a heated pan. They stir the eggs until they are cooked, and finally place the scrambled eggs onto a plate beside toast. These steps clearly indicate that the breakfast being prepared is scrambled eggs.</think>
Answer: scrambled eggs

---

Now, answer for the given video.""",


}

moment_retrieval_query_prompts = {
    "MomentSeeker": """You are given a query and a candidate video clip in a moment (video) retrieval task. The query can be:
- Text only
- Text plus an image
- Text plus query video

Your task is to reason about what the query implies, describe in detail what the target video clip should show, and finally provide a concise summary.

Instructions:
- Inside <think> ... </think>:
  Provide a detailed, factual description of the target clip based on the query, focusing on the scene, key objects, and actions.
- After </think>, on the next line, output a one-sentence summary starting with "Summary: ".

---

**EXAMPLE (for the text plus query video case)**

Query:  
Text: What happened to this window afterward?  
(Current video: A ball is shown flying quickly toward the window.)

<think> A ball smashes through the window, leaving the glass completely shattered. Pieces of glass are scattered across the floor near the sill, and the ball rests against the far wall inside the room. Sunlight streams through the broken frame, and small shards glitter on the floor.</think>
Summary: The window is shattered, with glass scattered on the floor and the ball inside the room.

---

Now, answer for the given query: {query}""",

"QVHighlight": """You are given a video and a query describing a specific clip in the video. Your task is to describe in detail only the video clip that matches the query, not the entire video.

Instructions:
- Inside <think> ... </think>:
  Provide a detailed, factual description of the localized clip, focusing on the subject, their actions, objects, and context.
- After </think>, on the next line, output a one-sentence summary starting with Summary: 

---

**EXAMPLE**

Query: person turn a light on  

<think>A person walks into a dimly lit room and heads toward the wall near the doorway. They raise their right hand and press the switch upward. Instantly, the room brightens, revealing a desk, a chair, and shelves against the wall. The person pauses for a second to glance around the room, confirming the light is on before stepping further inside.</think>
Answer: A person flips a switch, and the room lights up.

---

Now, answer for the given video with the following query: {query}""",
}



for dataset in VIDORE_QA_RETRIEVAL_DATASETS + VISRAG_QA_RETRIEVAL_DATASETS:
    prompts[dataset] = prompts["visrag-indomain"]

for dataset in ["MSVD", "MSR-VTT", "DiDeMo", "VATEX"]:
    prompts[dataset] = prompts["video_caption_300k"]

prompts['UCF101'] = prompts['Kinetics-700']
prompts['SmthSmthV2'] = prompts['HMDB51']

moment_retrieval_query_prompts['Charades-STA'] = moment_retrieval_query_prompts['QVHighlight']