#!/usr/bin/env python3
"""
Initialize weights (regularization)
"""
from tensorflow import keras


def build_model_initializer_by_activation(
        input_dim,
        hidden_units,
        activation
):
    """
    Use an appropriate weight initializer
    based on the activation function:
    'sigmoid', 'tanh', 'relu', or 'leaky_relu'
    returns a fucking shallow model
    """

    if not isinstance(
        (input_dim, hidden_units), int
    ):
        raise TypeError(
            'Wahtch out amounts of model features and neurons, baby'
            )
    if not isinstance(activation, str):
        raise TypeError('You need to name it a special word')

    if activation == 'sigmoid' or activation == 'tanh':
        initializer = keras.initializers.GlorotUniform()
    elif activation == 'relu' or activation == 'tanh':
        initializer = keras.initializers.HeNormal()
    else:
        raise ValueError('Check function docs for activation names')

    inputs = keras.layers.Input(shape=(input_dim,))
    hidden = keras.layers.Dense(
        1,
        units=hidden_units,
        activation=activation,
        kernel_initializer=initializer
        )(inputs)
    outputs = keras.layers.Dense(activation='softmax')(hidden)

    model = keras.Model(inputs, outputs)

    return model
