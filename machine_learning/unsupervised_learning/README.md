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
