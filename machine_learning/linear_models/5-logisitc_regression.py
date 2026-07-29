#!/usr/bin/env python3
"""
Dummy Logistric Regression Model
"""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """
    Performs binary classification
    by fitting logistic function
    """

    return linear_model.LogisticRegression(
        random_state=random_state
        )
