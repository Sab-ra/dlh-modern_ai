#!/usr/bin/env python3
"""
Find every 'blank' in a tokenized sentence.
"""


def get_mask_index(inputs, tokenizer):
    """
    Find the position index of every
    <mask> (50264) brick.
    Args:
        inputs: Tokenized text from the tokenizer.
        tokenizer: RoBERTa tokenizer that knows the blank.
    Returns:
        list[int]: Position indeces of 'blank's.
    """
    token_ids = inputs['input_ids'][0].tolist()
    mask_id = tokenizer.mask_token_id   # 50264 in this case
    mask_indices = [
        i for i, token_id in enumerate(token_ids)
        if token_id == mask_id
    ]
    if not mask_indices:
        raise ValueError(
            "No <mask> token found in the input!"
        )

    return mask_indices
