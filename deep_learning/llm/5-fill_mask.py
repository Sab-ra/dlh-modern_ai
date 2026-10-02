#!/usr/bin/env python3
"""
Ask the ready-made game to fill the blanks.
"""
import transformers


def fill_mask(model_name, top_k):
    """
    Build a helper that fills the blanks.
    Args:
        model_name (str): Wich RoBERTa to invite.
        top_k (int): How many favourite words
            to keep.
    Returns:
        The ready-made fill-in-the-blank helper.
    """
    fill = transformers.pipeline(
        'fill-mask',
        model=model_name,
        top_k=top_k
    )

    return fill
