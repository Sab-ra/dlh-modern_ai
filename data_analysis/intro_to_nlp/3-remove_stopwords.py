#!/usr/bin/env python3
"""
Stopwords removal for tokenized text messages.
"""
import nltk


def remove_stopwords(
        tokens,
        language='english',
        extra_words=None,
        keep_words=None
):
    """
    Filter stopwords out of a token list.
    """
    if not isinstance(tokens, list):
        return []

    stop_words = set(nltk.corpus.stopwords.words(language))

    if extra_words:
        stop_words |= set(extra_words)
    if keep_words:
        stop_words -= set(keep_words)

    return [t for t in tokens if t.lower() not in stop_words]
