#!/usr/bin/env python3
"""
Ask a spechalist model to
change language (translate).
"""
import transformers


def translate_text(
        model_name,
        src_lang=None,
        tgt_lang=None
):
    """
    Build a helper that translates sentences.
    Args:
        model_name (str): Which traduttorine
            to invite.
        src_lang (str): Translate from
        tgt_lang (str): Translate to
    Returns:
        The ready-made translation helper.
    """
    if src_lang is None:
        translator = transformers.pipeline(
            'translation',
            model=model_name
        )
    else:
        translator = transformers.pipeline(
            'translation',
            model=model_name,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )

    return translator
