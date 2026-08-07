#!/usr/bin/env python3
"""
Generate predictions using a trained model
"""
import tensorflow as tf


def predict(model, X, verbose=0):
    """
    Model, act my dear!
    Predict class labels for input data X
    """

    probabilities = model.predict(X, verbose=verbose)
    predicted_classes = tf.math.argmax(probabilities, axis=1)

    return predicted_classes.numpy()
