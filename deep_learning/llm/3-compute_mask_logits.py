#!/usr/bin/env python3
"""
Ask RoBERTa how strongly she feels
about each blank.
"""
import torch


def compute_mask_logits(model, inputs, mask_indices):
    """
    Read the sentence once and keep a score
    for every possible word at each blank/
    Args:
        model: RoBERTa, ready to answer.
        inputs: The row of cards,
        mask_indices (list): Where the blanks sit.
    Returns:
        list: One bundle of scores per blank.
    """
    with torch.no_grad():
        outputs = model(**inputs)

    all_scores = outputs.logits[0]
    mask_logits_list = []
    for index in mask_indices:
        mask_logits_list.append(all_scores[index])

    return mask_logits_list
