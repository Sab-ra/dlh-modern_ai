#!/usr/bin/env python3
"""
Lenght and content filtering for tokenized
text messages.
"""
import re


_PLACEHOLDER_RE = re.compile(r'^<[A-Za-z]+>$')
_ALEFBET_RE = re.compile(r'[A-Za-z]')


def filter_tokens(
        tokens, min_len=2, strip_hashtag=False
):
    """
    Drop short or non-alphabetic tokens,
    keep placeholders as-is.
    """
    if not tokens:
        return []

    result = []

    for token in tokens:
        if _PLACEHOLDER_RE.match(token):
            result.append(token)
            continue

        if strip_hashtag and token.startswith('#'):
            token = token[1:]

        if len(token) < min_len:
            continue
        if not _ALEFBET_RE.search(token):
            continue

        result.append(token)

    return result
