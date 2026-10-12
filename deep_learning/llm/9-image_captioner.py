#!/usr/bin/env python3
"""
Ask a model to write one sentence about a photo.
"""
import transformers
import PIL


def image_captioner(
    model,
    image_path,
    max_new_tokens
):
    """
    Write a short sentence about a photo.
    Args:
        model (str): Which ready-made caption model to invite.
        image_path (str): Where the photo is.
        max_new_tokens (int): How long the description may grow.
    Returns:
        A description of the photo, up to max tokens.
    """
    processor = transformers.BlipProcessor.from_pretrained(
        model
    )
    caption_model = transformers.BlipForConditionalGeneration.from_pretrained(
        model
    )

    image = PIL.Image.open(image_path). convert('RGB')
    inputs = processor(
        images=image, return_tensors='pt'
    )

    output = caption_model.generate(
        pixel_values=inputs['pixel_values'],
        max_new_tokens=max_new_tokens
    )
    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption
