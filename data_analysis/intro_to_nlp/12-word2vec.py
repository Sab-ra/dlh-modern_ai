#!/usr/bin/env python3
"""
Train Word2Vec and average in-vocab
vectors per text message.
"""
import numpy as np
import gensim.models


def word2vec_embeddings(
        corpus_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        sg=0,
        epochs=10,
        workers=4
):
    """
    Return (X, model): message embeddings and
    Word2Vec.
    """
    model = gensim.models.Word2Vec(
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
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if vecs:
            rows.append(np.mean(vecs, axis=0))
        else:
            rows.append(np.zeros(vector_size))
    X = np.vstack(rows).astype(np.float64)

    return X, model
