#!/usr/bin/env python3
"""
Surgically remove irrelevant features
- least,
- absolute,
- shrincage, and
- selection,
- operator
"""
from sklearn import linear_model


def lasso_regression(random_state):
    """
    Extends linear regression with
    L1 regularization (forsing least relevant beta's to zero)
    """

    return linear_model.Lasso(random_state=random_state)
