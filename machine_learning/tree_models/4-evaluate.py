#!/usr/bin/env python3
"""
Issue an evaluation report
on classifier quality
"""
from sklearn import metrics

def evaluate(
        true_labels,
        predicted_labels,
        class_names
):
    """
    Returns report for given labels
    """

    return metrics.classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )
