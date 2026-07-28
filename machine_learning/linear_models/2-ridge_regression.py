#!/usr/bin/env python3
"""
Regulaised L2 Model Creation
(ridge regression)
"""
from sklearn import linear_model


def ridge_regression(random_state):
    """
    Rounding out but never delete features
    """

    return linear_model.RidgeClassifier(random_state)
