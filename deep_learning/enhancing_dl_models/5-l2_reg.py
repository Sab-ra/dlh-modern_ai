#!/usr/bin/env python3
"""
Create Keras model with L2 regularization
"""
from tensorflow import keras


def build_model_with_L2_regularization(
        input_dim,
        hidden_units,
        n_layers,
        lambda_l2
):
    """
    Return Keras model with L2 regularization
    """

    for name, value in {
        'input_dim': input_dim,
        'hidden_units': hidden_units,
        'n_layers': n_layers
    }.items():
        if not isinstance(value, int):
            raise TypeError(
                f'{name} must be int, got {type(value).__name__}'
            )
    if not isinstance(lambda_l2, float):
        raise TypeError('Lambda L2 must be float')

    inputs = keras.layers.Input(shape=(input_dim,))
    hidden_layers = []
    for i in range(n_layers):
        if i == 0:
            hidden_layers.append(
                keras.layers.Dense(
                hidden_units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(lambda_l2)
                )(inputs)
            )
        else:
            hidden_layers.append(keras.layers.Dense(
                hidden_units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(lambda_l2)
                )(hidden_layers[i-1])
            )
    outputs = keras.layers.Dense(
        10,
        activation='softmax')(hidden_layers[n_layers-1])

    model = keras.Model(inputs, outputs)

    return model
