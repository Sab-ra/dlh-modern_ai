#!/usr/bin/env python3
"""
Compute feature importances in
a trained random forest
"""
import numpy as np


def feature_importance(rf):
    """
    Sort from least to most importance feature indeces
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    return importances, indices
