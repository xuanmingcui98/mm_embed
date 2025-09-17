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

HMDB51_LABELS = ['ride_bike', 'wave', 'dive', 'pour', 'smile', 'eat', 'shoot_ball', 'clap', 'chew', 'brush_hair', 'pushup', 'draw_sword', 'pullup', 'catch', 'somersault', 'sit', 'hug', 'kick_ball', 'golf', 'cartwheel', 'turn', 'kick', 'dribble', 'jump', 'sword_exercise', 'drink', 'shoot_gun', 'hit', 'ride_horse', 'smoke', 'fencing', 'climb', 'situp', 'handstand', 'talk', 'kiss', 'push', 'shake_hands', 'swing_baseball', 'stand', 'flic_flac', 'sword', 'run', 'laugh', 'walk', 'throw', 'punch', 'climb_stairs', 'pick', 'shoot_bow', 'fall_floor']
UCF101_LABELS = ['Kayaking', 'PlayingDaf', 'BoxingSpeedBag', 'GolfSwing', 'Diving', 'PlayingFlute', 'TrampolineJumping', 'BabyCrawling', 'Drumming', 'RockClimbingIndoor', 'Biking', 'MoppingFloor', 'Haircut', 'SalsaSpin', 'PlayingPiano', 'HandstandWalking', 'Billiards', 'Nunchucks', 'SoccerJuggling', 'VolleyballSpiking', 'Skijet', 'LongJump', 'UnevenBars', 'ApplyEyeMakeup', 'SoccerPenalty', 'BandMarching', 'BlowingCandles', 'PizzaTossing', 'ApplyLipstick', 'HandstandPushups', 'CricketShot', 'FrisbeeCatch', 'PushUps', 'FieldHockeyPenalty', 'ThrowDiscus', 'PlayingGuitar', 'JugglingBalls', 'HorseRiding', 'PlayingViolin', 'JumpingJack', 'PoleVault', 'BoxingPunchingBag', 'FloorGymnastics', 'RopeClimbing', 'JumpRope', 'ParallelBars', 'BodyWeightSquats', 'HorseRace', 'BasketballDunk', 'BlowDryHair', 'SkyDiving', 'BalanceBeam', 'IceDancing', 'SumoWrestling', 'Bowling', 'TennisSwing', 'MilitaryParade', 'Lunges', 'Swing', 'HighJump', 'StillRings', 'Skiing', 'HeadMassage', 'JavelinThrow', 'HammerThrow', 'Hammering', 'BaseballPitch', 'Shotput', 'Basketball', 'PommelHorse', 'Punch', 'TableTennisShot', 'SkateBoarding', 'Typing', 'Rafting', 'WritingOnBoard', 'PlayingSitar', 'Archery', 'PlayingTabla', 'TaiChi', 'BreastStroke', 'ShavingBeard', 'CricketBowling', 'Rowing', 'CliffDiving', 'CleanAndJerk', 'PullUps', 'PlayingDhol', 'YoYo', 'FrontCrawl', 'WallPushups', 'WalkingWithDog', 'Knitting', 'Mixing', 'HulaHoop', 'BrushingTeeth', 'Surfing', 'PlayingCello', 'BenchPress', 'CuttingInKitchen', 'Fencing']
BREAKFAST_LABELS = ['pancake', 'cereal', 'sandwich', 'scrambledegg', 'friedegg', 'coffee', 'milk', 'tea', 'juice', 'salad']

query_prompt_template = {
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
- Follow the examples below for guidance.

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



"MVBench": """You are given a video and a multiple-choice question about it. Please select the correct option from the provided choices following the below instructions. Let's think step by step.

Instructions:
- Wrap your reasoning inside <think> ... </think> tags.
- Keep your reasoning succinct (2–3 sentences).
- On a new line, output the final answer chosen from the provided options in this exact format: Answer: <option>

---

EXAMPLE:

Question: Why did Castle dress like a fairy when he was speaking to Emily?  
Options:
(A) To get her to trust him
(B) He secretly loved fairies
(C) He lost a bet with Emily

<think>The video shows Castle walking into a classroom wearing a pink fairy costume with wings and a wand. Emily looks upset and hesitant to talk to him at first. Castle kneels down to her level, speaks gently, and smiles reassuringly. After a moment, Emily starts talking and visibly relaxes. None of the other children are dressed similarly, ruling out option D. There’s no mention or visual cue of a bet, a school event, or Mrs. Ruiz forcing him. The interaction and Castle’s behavior strongly suggest he dressed up intentionally to make Emily comfortable and earn her trust.</think>
Answer: (A) To get her to trust him

---

Now, answer for the given video with the following question:

Question: {query}""",

"Video-MME": """You are given a video and a multiple-choice question about it. Please select the correct option from the provided choices following the below instructions. Let's think step by step.

Instructions:
- Wrap your reasoning inside <think> ... </think> tags.
- Keep your reasoning succinct (2–3 sentences).
- On a new line, output the final answer chosen from the provided options in this exact format: Answer: <option>

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

"NExTQA": """You are given a video and a multiple-choice question about it. Please select the correct option from the provided choices following the below instructions. Let's think step by step.

Instructions:
- Wrap your reasoning inside <think> ... </think> tags.
- Keep your reasoning succinct (2–3 sentences).
- On a new line, output the final answer chosen from the provided options in this exact format: Answer: <option>

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

"ActivityNetQA": """You are given a video and a yes/no question about it. Please answer "yes" or "no" following the below instructions. Let's think step by step.

Instructions:
- Wrap the description and reasoning inside <think> ... </think> tags.
- Keep your reasoning succinct (2–3 sentences).
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

"EgoSchema": """You are given a video and a multiple-choice question about it. Please select the correct option from the provided choices following the below instructions. Let's think step by step.

Instructions:
- Wrap your reasoning inside <think> ... </think> tags.
- Keep your reasoning succinct (2–3 sentences).
- On a new line, output the final answer chosen from the provided options in this exact format: Answer: <option>

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

"HMDB51": f"""You are given a video showing a person performing an action from the HMDB51 dataset. Your task is to first generate a detailed description of the video content and your reasoning process to identify the action, and finally provide the answer in the specified format. Let's think step by step.

Instructions:
- Wrap the reasoning inside <think> ... </think> tags.
- On a new line, output the final answer in this exact format:  
  `Answer: <action word or short phrase>`
- Choose the action from the below HMDB51 action list:

{HMDB51_LABELS}

---

**EXAMPLE**

Question: What action is the person performing in the video?  

<think>The video shows a man standing in front of a punching bag. He repeatedly throws quick punches using both hands, alternating between left and right. His stance and arm movements are consistent with practicing boxing techniques.</think>
Answer: boxing

---

Now, answer for the given video.""",

"UCF101": f"""You are given a video showing a person performing an action from the UCF101 dataset. Your task is to first generate a detailed description of the video content and your reasoning process to identify the action, and finally provide the answer in the specified format. Let's think step by step.

Instructions:
- Wrap the reasoning inside <think> ... </think> tags.
- Keep your reasoning succinct (2–3 sentences).
- On a new line, output the final answer in this exact format:  
  `Answer: answer
- Choose the action from the below UCF101 action list:

{UCF101_LABELS}

---

**EXAMPLE**

Question: What action is the person performing in the video?  

<think>The video shows a man standing in front of a punching bag. He repeatedly throws quick punches using both hands, alternating between left and right. His stance and arm movements are consistent with practicing boxing techniques.</think>
Answer: boxing

---

Now, answer for the given video.""",

"Kinetics-700": """You are given a video from Kinetics-700 dataset, and your task is to recognize the category of the video content. Let's think step by step.

Instructions:
- Wrap the reasoning inside <think> ... </think> tags.
- On a new line, output the final answer in this exact format:
  `Answer: <kinetics-700_category>`
- The category should be one of the 700 predefined Kinetics-700 classes.

---

EXAMPLE

Question: What is the category of the video content?  

<think>
The video shows a person standing at a kitchen counter with vegetables laid out. They pick up a knife and carefully slice tomatoes, onions, and peppers, placing the pieces into a bowl. The consistent cutting and preparation of food ingredients indicate that the action being performed is chopping vegetables.</think>
Answer: chopping_vetegables

---

Now, answer for the given video.""",

"Breakfast": f"""You are given a video and a list of breakfast type. Your task is to recognize the type of breakfast the person is preparing from the predefined breakfast list. Let's think step by step.

Instructions:
- Wrap the reasoning inside <think> ... </think> tags.
- On a new line, output the final answer in this exact format:
  `Answer: <breakfast type>`
- Choose the breakfast type from the below list:

{BREAKFAST_LABELS}

---

**EXAMPLE**

Question: What type of breakfast is the person preparing in the video?  

<think>
The video shows a person cracking eggs into a bowl, whisking them, and then pouring the mixture into a heated pan. They stir the eggs until they are cooked, and finally place the scrambled eggs onto a plate beside toast. These steps clearly indicate that the breakfast being prepared is scrambled eggs.</think>
Answer: scrambledegg

---

Now, answer for the given video.""",

"SmthSmthV2": """You are given a video showing a person performing one or more actions or interacting with objects. Your task is to identify the action or object interaction being performed.

Instructions:
- Inside <think> ... </think>:
  1. Describe the relevant parts of the clip in detail, focusing on the person’s movements, the objects they handle, and the context around the interaction.
  2. Be objective and precise, avoiding speculation or stylistic commentary.
- After </think>, output the action or interaction as a short phrase in this exact format:
  `Answer: <action or interaction>`
- Follow the examples below for guidance.

---

Query: What actions or object interactions are being performed by the person in the video?

**EXAMPLE**

<think>
The clip shows a person seated at a desk with a thin white paper sleeve in front of them. They pick up the sleeve with one hand and steady it with the other, gently pinching the edges to open it wider. With a careful, smooth motion, they slide a shiny disc halfway out, tilting it slightly as light reflects off the surface. They handle the disc by its edges to avoid touching the shiny side, then fully remove it and place it neatly on the desk next to a small stack of similar discs.
</think>  
Answer: Pulling a disc out of a paper sleeve.

---

Now, answer for the given video.""",

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
  Provide a detailed, factual description of the localized clip.
- After </think>, on the next line, output a one-sentence summary starting with Summary: 

---

**EXAMPLE**

Query: person turn a light on  

<think>A person walks into a dimly lit room and heads toward the wall near the doorway. They raise their right hand and press the switch upward. Instantly, the room brightens, revealing a desk, a chair, and shelves against the wall. The person pauses for a second to glance around the room, confirming the light is on before stepping further inside.</think>
Answer: A person flips a switch, and the room lights up.

---

Now, answer for the given video with the following query: {query}""",
}

query_prompt_no_cot = {
    "Breakfast": f"""You are given a video and a list of breakfast type. Your task is to recognize the type of breakfast the person is preparing from the predefined breakfast list:

{BREAKFAST_LABELS}

Only provide the final answer in your response. Do not include any explanations or reasoning.""",

"SmthSmthV2": """You are given a video from SomethingSomethingV2 dataset showing a person performing one or more actions or interacting with objects. Your task is to identify the action or object interaction being performed. Provide the final answer as a short sentence without any explanation.""",

"Video-MME": """Question: {query}\n\nAnswer with the option from the given choices directly and only give the best option.""",

"NExTQA": """Question: {query}\n\nAnswer with the option from the given choices directly and only give the best option.""",

"EgoSchema": """Question: {query}\n\nAnswer with the option from the given choices directly and only give the best option.""",

"HMDB51": f"""Question: What is the primary action being performed? Choose the action from the below action list:{HMDB51_LABELS}""",

"ActivityNetQA": """Question: {query}\nAnswer with "yes" or "no" directly.""",

"MVBench": """Question: {query}\n\nAnswer with the option from the given choices directly and only give the best option.""",

"Kinetics-700": """You are given a video from Kinetics-700 dataset, and your task is to recognize the category of the video content. The category should be one of the 700 predefined Kinetics-700 classes.

Only provide the final answer in your response. Do not include any explanations or reasoning.""",

"UCF101": f"""You are given a video showing a person performing an action from the UCF101 dataset. Your task is to identify the action being performed. Choose the action from the below UCF101 action list:
{UCF101_LABELS}

Only provide the final answer in your response. Do not include any explanations or reasoning.""",
    
}

target_prompt_template = {
    "visrag-indomain":
"""You are given a document image. Your task is to generate a concise factual description based only on the visible content.

Instructions:

- Extract facts only from the document — do not make assumptions or add external information.
- Write the description in 3–4 sentences
- Avoid subjective opinions, speculation, or stylistic commentary.
- Wrap the description and reasoning inside <think> ... </think> tags.
- On a new line, output a one-sentence summary in this exact format: Answer: <answer>
- Follow the examples below for guidance.

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
- Follow the examples below for guidance.

---

EXAMPLE:

<think>The video features a medium shot of an older woman with short white hair and fair skin, wearing a white shirt and a purple floral jacket. She speaks directly to the camera, occasionally looking to the right. The setting includes a stone fireplace with a black grate and tools, and a doorway in the background. A small black microphone is clipped to her jacket. The handheld camera gradually zooms in on her as she talks.</think>
Summary: The video shows an older woman speaking to the camera in front of a fireplace as the camera zooms in.

---

Now, generate a detailed description for the given video.""",

    "YouCook2":
"""You are given a cooking video. Your task is to generate a thorough and detailed description of the cooking activity.

Instructions:
- Focus on the cooking actions, ingredients, and interactions visible in the video.
- Describe the temporal sequence of steps clearly.
- Avoid subjective opinions, speculation, or stylistic commentary.
- Wrap the detailed description inside <think> ... </think> tags.
- On a new line, output a short summary of the video in this exact format:
Summary: <summary>

EXAMPLE:

<think>The video shows a person standing at a kitchen counter with a glass mixing bowl. They squeeze lemon juice into the bowl, then add a spoonful of reddish sumac and a small amount of finely minced garlic. A pinch of salt is sprinkled in, followed by a steady stream of oil. All the ingredients collect together in the bowl, ready to be stirred into a dressing.</think>
Summary: combine lemon juice sumac garlic salt and oil in a bowl


Now, generate a detailed description for the given video.""",


    "DiDeMo":
"""You are given a video. Your task is to generate a detailed description by grouping the video into temporal event segments, and providing one description for each segment.

Instructions:
- Provide a detailed description of the video by breaking the video into meaningful temporal events.
- List the descriptions in temporal order.
- Wrap the detailed description inside <think> ... </think> tags.
- On a new line, output a concise summary of the entire video in this exact format:
Summary: <summary>
- Follow the examples below for guidance.

---

**EXAMPLE**

<think>The black cat enters the frame and pauses near the doorway. The black cat turns to look at the camera. An orange-and-white cat enters and walks toward the black cat. Both cats interact briefly, sniffing each other. Both cats walk toward the couch and sit together near the window.</think>
Summary: the cat looks back at the camera. black cat first appears black cat joins orange and white cat the cat is first visible.

---

Now, generate detailed descriptions for the given video.""",

    "VATEX": """You are given a video. Your task is to generate a thorough and detailed description that describes the video content.

Instructions:
- Wrap the description inside <think> ... </think> tags.
- On a new line, output a short summary in this exact format: Summary: <summary>

---

EXAMPLE:

<think>The video shows a young boy sitting outdoors on a bench, his hair and clothes moving in the wind. He holds a small stack of trading cards in his hands and tilts them toward the camera, displaying different designs on the cards. After showing several cards, he gathers them back together and demonstrates shuffling the deck with both hands. Leaves and grass can be seen moving in the breeze around him, emphasizing the windy setting.</think>
Summary: A boy sits outside in the wind, shows his stack of trading cards to the camera, and then shuffles them in his hands.

---

Now, generate a detailed description for the given video.""",
}





for dataset in VIDORE_QA_RETRIEVAL_DATASETS + VISRAG_QA_RETRIEVAL_DATASETS:
    target_prompt_template[dataset] = target_prompt_template["visrag-indomain"]

for dataset in ["MSVD", "MSR-VTT"]:
    target_prompt_template[dataset] = target_prompt_template["video_caption_300k"]


query_prompt_template['Charades-STA'] = query_prompt_template['QVHighlight']

for dataset in ['MomentSeeker', 'QVHighlight', 'Charades-STA']:
    target_prompt_template[dataset] = target_prompt_template["DiDeMo"]