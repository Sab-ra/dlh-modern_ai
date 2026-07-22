#!/usr/bin/env python3
"""
Generate predictions out of
a trained tree-based classifier
"""


def generate_predictions(clf, X):
    """
    Return predictions per input data
    """

    return clf.predict(X)
