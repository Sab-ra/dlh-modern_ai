#!/usr/bin/env python3
"""
Shallow model (one layer of neirones)
Performs multi-class classification
"""
from tensorflow import keras


def build_model(input_dim):
    """
    Use Sequential class
    """

    return keras.Model(input_dim=input_dim, neurons_h=1)
