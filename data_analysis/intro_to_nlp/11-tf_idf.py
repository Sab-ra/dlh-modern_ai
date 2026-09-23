#!/usr/bin/env python3
"""
TF-IDF feature matrix from preprocessed token lists.
"""
import sklearn


def tf_idf(
        courpus_tokens,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        norm='l2'
):
    """
    Fit TfidfVectorizer; return (X, vectorizer).
    """
    docs = [' '.join(tokens) for tokens in courpus_tokens]
    vectorizer = sklearn.feature_extraction.text.TfidVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        norm=norm
    )
    X = vectorizer.fit_transform(docs)

    return X, vectorizer
