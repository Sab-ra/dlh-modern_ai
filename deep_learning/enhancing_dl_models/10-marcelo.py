#!/usr/bin/env python3
"""Enhancing Deep Learning Models.
This module provides functions for optimization, regularization,
and hyperparameter tuning in Keras."""
import keras_tuner


def search_and_return_best_model(
        tuner, x_train, y_train, epochs, validation_split, verbose=0):
    """Executes a KerasTuner search and returns the best
    hyperparameter configuration."""
    tuner.search(x_train, y_train, epochs=epochs,
                 validation_split=validation_split, verbose=verbose)
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hyperparameters