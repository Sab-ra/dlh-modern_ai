#!/usr/bin/env python3
"""
Set Demilitarized Zone with SVM
"""
from sklearn import svm


def get_SVM_model(name, random_state):
    """
    Accepts names of kernels:
    linear, poly, and rbf for radial basis function
    """

    return svm.SVC(
        kernel=name,
        random_state=random_state
        )
