#!/usr/bin/env python3
"""
Build functional shallow newral network
"""
from tensorflow import keras


def build_model(input_dim, neurons_h):
    """
    Perform multiclass classification
    without using the Sequential class
    """

    model = keras.Functional(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(neurons_h, activation='sigmoid'),
            keras.layers.Dense(10, activation='softmax')
        ]
    )

    return model
