#!/usr/bin/env python3
"""
Evaluate K-Means clustering for different numbers of clusters
"""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """
    Returen cluster counts,
    inertia values, and
    silhouette scores
    """

    cluster_numbers = []
    inertia_values = []
    silhouette_scores = []

    for n_clusters in range(2, max_clusters+1):
        k_means = K_Means(X, n_clusters, random_state)
        labels = k_means.labels_

        cluster_numbers.append(n_clusters)
        inertia_values.append(k_means.inertia_)
        silhouette_scores.append(
            metrics.silhouette_score(X, labels)
        )

    return cluster_numbers, inertia_values, silhouette_scores
