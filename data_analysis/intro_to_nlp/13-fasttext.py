#!/usr/bin/env python3
"""
Train FastText and average token vectors
per message.
"""
import numpy as np
import gensim.models


def fasttext_embeddings(
        corpus_tokens,
        vector_size=100,
        window=5,
        min_count=1,
        sg=0,
        epochs=10,
        workers=4
):
    """
    Return (X, model): message embeddings and FastText.
    """
    model = gensim.models.FastText(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers
    )

    rows = []
    for tokens in corpus_tokens:
        if not tokens:
            rows.append(np.zeros(vector_size))
            continue
        vecs = [model.wv[t] for t in tokens]
        rows.append(np.mean(vecs, axis=0))
    X=np.vstack(rows).astype(np.float64)

    return X, model
