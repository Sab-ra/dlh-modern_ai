#!/usr/bin/env python3
"""
Pull a awift brain and preserve it's knowledge
"""
import matplotlib.pyplot as plt
import seaborn as sns


def build_feature_extractor():
    """
    Loads MobileNetV2 without top layer,
    freezes its weights, and
    attaches a globalAveragePooling2D layer on top.
    """

    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D()
    ])

    return model
