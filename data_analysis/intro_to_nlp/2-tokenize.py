#!/usr/bin/env python3
"""
Tokenization of cleaned text messages.
"""
import nltk


EMOTICON_MAP = {
    "<3": "<EMO>", "</3": "<EMO>",
    ":)": "<EMO>", ":-)": "<EMO>",
    ":(": "<EMO>", ":-(": "<EMO>",
    ":d": "<EMO>", ";)": "<EMO>",
    ":|": "<EMO>", ">:(": "<EMO>",
    ":p": "<EMO>", "b)": "<EMO>",
    "o:)": "<EMO>"
}

_TWEET_TOKENIZER = nltk.tokenize.TweetTokenizer(
    reduce_len=True
)


def normalize_emoticons(
        tokens,
        emoticon_action='replace'
):
    """
    Map known emoticon tokens to a shared placeholder.
    """
    if not isinstance(tokens, list):
        return []

    result = []

    for token in tokens:
        mapped = EMOTICON_MAP.get(token.lower())

        if mapped:
            if emoticon_action == 'replace':
                result.append(mapped)
        else:
            result.append(token)

    return result


def tokenize_text(text, method='tweet'):
    """
    Split a cleaned message into tokens.
    """
    if not isinstance(text, str):
        return []

    if method == 'tweet':
        return _TWEET_TOKENIZER.tokenize(text)
    if method == 'word':
        return nltk.word_tokenize(text)
    if method == 'split':
        return text.split()

    raise ValueError('Invalid tokenizer method')
