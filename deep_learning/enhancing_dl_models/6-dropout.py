#!/usr/bin/env python3
"""
Regularisation by dropping neurones
"""
from tensorflow import keras


def build_model_with_dropout(
        input_dim,
        hidden_units,
        n_layers,
        dropout_rate_input,
        dropout_rate_hidden
):
    """
    Returns a keras model with dropout regularisation
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

    for name, value in {
        'dropout_rate_input': dropout_rate_input,
        'dropout_rate_hidden': dropout_rate_hidden
    }.items():
        if not isinstance(value, (float, int)):
            raise TypeError(
                f'{name} must be float, got {type(value).__name__}'
            )
    dropout_rate_input = float(dropout_rate_input)
    dropout_rate_hidden = float(dropout_rate_hidden)

    inputs = keras.layers.Input(
        shape=(input_dim,),
        kernel_regularizer=keras.regularizers.dropout(
            dropout_rate=dropout_rate_input
            )
        )
    x = inputs
    for _ in range(n_layers):
        x = keras.layers.Dense(
            hidden_units,
            activation='relu',
            kernel_regularizer=keras.regularizers.dropout(
                dropout_rate=dropout_rate_hidden
            )
        )(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    return model
