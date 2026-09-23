#!/usr/bin/env python3
"""
Bag-of-Words matrix from preprocessed token lists.
"""
import sklearn


def bag_of_words(
    corpus_tokens,
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    binary=False
):
    """
    Fit CountVectorizer; return (X, vectorizer).
    """
    docs = [' '.join(tokens) for tokens in corpus_tokens]
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary
    )
    X = vectorizer.fit_transform(docs)

    return X, vectorizer
