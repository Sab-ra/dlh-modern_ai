#!/usr/bin/env python3
"""
Judges score the scribe's guesses.
"""
import numpy as np
import sklearn.metrics

def compute_metrics(predictions):
    """
    Return weighted accuracy, precision,
    recall, and f1.
    """
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = (
        sklearn.metrics.precision_recall_fscore_support(
            labels=labels,
            preds=preds,
            average='weighted'
        )
    )

    return {
        'accuracy': sklearn.metrics.accuracy_score(labels, preds),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
