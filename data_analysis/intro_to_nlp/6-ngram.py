#!/usr/bin/env python3
"""
N-gram generation from a token list.
"""
import nltk


def generate_ngrams(tokens, n=2):
    """
    Join each n consecutive tokens with an underscore.
    """
    if not isinstance(token, list) or len(tokens) < n:
        return []

    return [
        '_'.join(gram)
        for gram in nltk.ngrams(tokens, n)
    ]
