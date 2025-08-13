
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

<think>The document lists the top 20 most spoken languages by native speakers, led by Chinese (Mandarin) with 935 million, followed by Spanish (387M), English (365M), Hindi (295M), and Arabic (280M). A pie chart shows each language’s share of the world population, with Chinese at 14.1% and “Other” languages totaling 46.45%. Geographic regions for each language are shown, such as Mandarin in China, Taiwan, and Singapore, and Spanish in Latin America and Spain. Greetings in each language are also displayed.</think>
Answer: The document ranks the top 20 most spoken languages, shows their population share, and maps their primary regions of use.

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
- On a new line, output a summary in this exact format: Answer: <answer>

---

EXAMPLE:

<think>The video opens with a view of an indoor court marked with various lines, presumably for different sports. There are four red balls aligned on the center line. In subsequent frames, people enter the frame, and a dodgeball game commences. The players are seen throwing balls at one another, maneuvering around the court to avoid being hit, and attempting to catch the balls. The participants vary in their attire, indicating a casual and informal game setting. The color contrasts of the red balls and the colored lines on the floor are visually prominent throughout the video. As the game progresses, players are occasionally hit by a ball and step out of the main play area. There appears to be good-natured interaction between the players, supportive of a friendly game rather than a competitive match.</think>
Answer: An indoor dodgeball game with players dodging and throwing red balls.

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
Video: <video>"""}