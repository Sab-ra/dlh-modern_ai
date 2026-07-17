# Unsupervised Machine Learning

## 0-standardize.py

### Feature Standardization

Write a function Standardize(X) that standardizes tabular data using Scikit-learn.

This preprocessing step is essential for distance-based and variance-based unsupervised learning techniques, ensuring that all features contribute proportionally.

- Arguments:
X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
- Returns:
numpy.ndarray: The standardized version of the input data, with the same shape as X.

_Required import: `from sklearn import preprocessing`_.

## 1-pca.py

### Dimensionality Reduction with PCA

Write a function `Apply_PCA(X, n_components, random_state)` that performs Principal Component Analysis (PCA) on tabular data using Scikit-learn.

Arguments:
- X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
- n_components (int | float | None):
- int: Number of principal components to keep
- float (between 0 and 1): Minimum fraction of variance to preserve
- None: Keep all components
- random_state (int): Random seed for reproducibility

Returns:
- numpy.ndarray: Data transformed into principal component space.
- sklearn.decomposition.PCA: Fitted PCA instance.

_Required import: `from sklearn import decomposition`._

**Full PCA: Transform All Features & Show All Components**

## 2-k_means.py

### Clustering with K-Means

Write a function `K_Means(X, n_clusters, random_state)` that creates and fits a K-Means clustering model using Scikit-learn on tabular data.

Arguments:
- X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
- n_clusters (int): Number of clusters to find
- random_state (int): Random seed for reproducibility
Rturns:
- sklearn.cluster.KMeans: K-Means instance fitted on the input data.

_Required import: `from sklearn import cluster`._

**K-Means Clustering in 2D PCA Space**

## 3-optimal_k.py

### Choosing the Optimal K for K-Means

Write a function `optimal_k(X, max_clusters, random_state)` that evaluates K-Means clustering quality using silhouette scores to measure cluster cohesion and separation, and to compute the inertia to be used for the elbow method.

Arguments:
- X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
- max_clusters (int): Maximum number of clusters to evaluate (>=2)
- random_state (int): Random seed for reproducibility

Returns:
- list[int]: Evaluated cluster numbers
- list[float]: Inertia values corresponding to each cluster number for the elbow method
- list[float]: Silhouette scores corresponding to each cluster number for cluster quality evaluation

Required import:
- from sklearn import metrics
- K_Means = __import__('2-k_means').K_Means.

**K-Means Clustering for Various k Values**

## 4-agglomerative.py

### Agglomerative Hierarchical Clustering

Write a function `Agglomerative_Clustering(X, n_clusters, random_state, n_components, use_pca_data=True)` that performs Agglomerative hierarchical clustering on tabular data using Scikit-learn.

The function will perform three main tasks:

1. Dimensionality reduction (optional): Apply PCA to reduce the data to n_components dimensions if use_pca_data=True.
1. Clustering: Fit an Agglomerative Clustering model on the (PCA-reduced or original) data using Ward linkage.
1. Evaluation: Compute the silhouette score for the clustering (if n_clusters > 1).

Arguments:
- X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
- n_clusters (int): Number of clusters
- random_state (int): Random seed for reproducibility
- n_components (int): Number of PCA components to reduce the data to (used only if use_pca_data=True)
- use_pca_data (bool, default=True): Whether to apply PCA to reduce dimensionality before clustering

Returns:
- sklearn.cluster.AgglomerativeClustering: Fitted Agglomerative Clustering instance
- numpy.ndarray: Data used for fitting (PCA-reduced or original)
- float: Silhouette score of the clustering (None if n_clusters=1)

_Required import: `from sklearn import cluster`, `from sklearn import metrics`, `Apply_PCA = __import__('1-pca').Apply_PCA`._

**Agglomerative Clustering Results on PCA-Reduced and Original Wine Data**
