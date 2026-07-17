#!/usr/bin/env python3
"""
Standartize tabular data
"""
from sklearn import preprocessing

def standardize(X):
    """
    Use StandardScaler form Scikit learn
    """

    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
