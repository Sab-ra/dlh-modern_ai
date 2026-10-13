#!/usr/bin/env python3
"""
Fetch the Emotion letter chest from the Hugging
Face hub.
"""
from datasets import load_dataset


def load_emotion_dataset():
    """
    Load dair-ai/emotion with its train,
    validation, and test splits.
    """
    return load_dataset('dair-ai/emotion')
