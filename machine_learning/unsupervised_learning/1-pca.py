#!/usr/bin/env python3
"""
Apply Principal Component Analysis (PCA)
to tabular data
"""
from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """
    Fit PCA on X and return transformed data
    and thee fitted PCA instance.
    """

    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )

    transformed = pca.fit_transform(X)

    return transformed, pca
