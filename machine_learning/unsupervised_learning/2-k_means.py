#!/usr/bin/env python3
"""
Fit a K-Means clustering model on tabular data.
"""
from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """
    Create and fit a K-Means model,
    then return the fitted instance
    """

    k_means = cluster.KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )

    k_means.fit(X)

    return k_means