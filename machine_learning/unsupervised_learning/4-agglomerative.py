#!/usr/bin/env python3
"""
Perform agglomerative hierarchical clustering on tabular data.
"""
from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(
        X,
        n_clusters,
        random_state,
        n_components,
        use_pca_data=True
):
    """
    Fit agglomerative clustering and return the model, data, and score.
    """

    if use_pca_data:
        data_used, _ = Apply_PCA(X, n_components, random_state)
    else:
        data_used = X

    model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    fitted_model = model.fit(data_used)

    if n_clusters > 1:
        score = metrics.silhouette_score(data_used, fitted_model.labels_)
    else:
        score = None

    return fitted_model, data_used, score
