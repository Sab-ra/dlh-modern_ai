#!/usr/bin/env python3
"""
Turn each score back into a readable word.
"""


def decode_mask_predictions(
    mask_logits_list,
    tokenizer
):
    """
    Name every slip in each drawer.
    Args:
        mask_logits_list (list): One drawer of
            scores for each blank.
        tokenizer: Knows which number is which word.
    Returns:
        list: For each blank, every word she knows.
    """
    decoded_tokens = []
    for logits in mask_logits_list:
        words = []
        for i in range(logits.shape[0]):
            word = tokenizer.decode([i]).strip()
            words.append(word)
        decoded_tokens.append(words)

    return decoded_tokens
