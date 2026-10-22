#!/usr/bin/env python3
"""
Fetch the Emotion letter chest from the Hugging
Face hub.
"""
import datasets


def load_emotion_dataset():
    """
    Load dair-ai/emotion with its train,
    validation, and test splits.
    """
    return datasets.load_dataset('dair-ai/emotion')
