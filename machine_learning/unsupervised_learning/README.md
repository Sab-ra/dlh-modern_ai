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
