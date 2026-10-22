#!/usr/bin/env python3
"""
The Feeling Teller booth in the market square.
"""
import transformers


def inference_mode(model_path, top_k):
    """
    Return a text-classification pipeline
    for the trained scribe.
    """
    return transformers.pipeline(
        task='text-classification',
        model=model_path,
        tokenizer=model_path,
        top_k=top_k
    )
