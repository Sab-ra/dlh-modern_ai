#!/usr/bin/env python3
"""
Cut text into numbered pieces for RoBERTa
"""
import transformers


def tokenize_text(
    model_name,
    sentence,
    padding=True
):
    """
    Load a RoBERTa tokenizer and cut text
    into pieces
    Args:
        model_name (str): Name of the pretr model
        sentence (str): Text to cut into pieces
        padding (bool): Add empty cushions, so rows match
    Returns:
        tokenizer (RobertaTokenizer): The piece-cutter
        inputs: Numbered pieces as PyTorch tensors
    """
    tokenizer = (
        transformers.RobertaTokenizer.from_pretrained(
            model_name
        )
    )
    inputs = tokenizer(
        sentence,
        padding=padding,
        return_tensors='pt'
    )

    return tokenizer, inputs
