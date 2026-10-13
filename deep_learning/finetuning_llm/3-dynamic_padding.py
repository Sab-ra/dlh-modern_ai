#!/usr/bin/env python3
"""
Make each bench only as wide as the
longest phrase sitting on it.
"""
import transformers


def create_data_collator(tokenizer):
    """
    Sets up dynamic padding for
    tokenized inputs using a
    pretrained tokenizer.
    """
    return transformers.DataCollatorWithPadding(
        tokenizer=tokenizer
    )
