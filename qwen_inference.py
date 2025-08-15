from PIL import Image

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image

def batch_inference(model, processor, texts=None, images=None, **kwargs):
    texts = [t.replace("<image>", "").strip() for t in texts] if texts is not None else [None] * len(images)
    # Preparation for batch inference
    images = [process_image(Image.open(image).convert("RGB"), resolution='high') if isinstance(image, str) else image for image in images] if images is not None else [None] * len(texts)
    messages = [
        processor.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": msg}]}], tokenize=False, add_generation_prompt=True) if img is not None else
        processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": msg}]}], tokenize=False, add_generation_prompt=True)
        for img, msg in zip(images, texts)
    ]

    images = images if not all(img is None for img in images) else None

    inputs = processor(
        text=messages,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=768, **kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts