#!/usr/bin/env python3
"""
Stemming and POS-aware lemmatization of tokens.
"""
import re
import nltk


_PLACEHOLDER_RE = re.compile(r'^<[A-Za-z]+>$')

_POS_MAP = {
    'J': nltk.corpus.wordnet.ADJ,
    'V': nltk.corpus.wordnet.VERB,
    'N': nltk.corpus.wordnet.NOUN,
    'R': nltk.corpus.wordnet.ADV
}


def get_pos(tag):
    """
    Map a Penn Treebank POS tag to
    a WordNet POS tag.
    """
    return _POS_MAP.get(tag[0], nltk.corpus.wordnet.NOUN)


def normalize_tokens(tokens, method='lemmatize'):
    """
    Reduce tokens to a base form,
    leaving placeholders untouched.
    """
    if method not in ('lemmatize', 'stem'):
        raise ValueError(
            "method must be 'lemmatize' or 'stem'"
        )

    if method == 'stem':
        stemmer = nltk.stem.PorterStemmer()
        return [
            token if _PLACEHOLDER_RE.match(token)
            else stemmer.stem(token)
            for token in tokens
        ]

    lemmatizer = nltk.stem.WordNetLemmatizer()
    tagged = nltk.pos_tag(tokens)
    return [
        token if _PLACEHOLDER_RE.match(token)
        else lemmatizer.lemmatize(token, pos=get_pos(tag))
        for token, tag in tagged
    ]
