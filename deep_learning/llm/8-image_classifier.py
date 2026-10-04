#!/usr/bin/env python3
"""
Invite a vision model that can behave as language.
"""
import transformers


def image_classifier(model):
    """
    Build a ready-made image classifier.
    The pipeline loads the pretrained vision model,
    prepares each photo, and returns the most likely
    labels with scores.
    Args:
        model (str): Name of the pretrained model to invite,
            for example 'google/vit-base-patch16-224'.
    Returns:
        A huggin face image-classification pipeline.
    """
    classifier = transformers.pipeline(
        'image-classification',
        model=model
    )

    return classifier
