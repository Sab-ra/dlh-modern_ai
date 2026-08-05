#!/usr/bin/env python3
"""
Train keras model
"""
from tensorflow import keras


def train_model(model, X, Y, epochs, verbose=1):
    """
    Buy your bitch fitness membourship
    """
    model.fit(
        X, Y,
        epochs=epochs,
        verbose=verbose
    )
