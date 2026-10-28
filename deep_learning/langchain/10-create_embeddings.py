#!/usr/bin/env python3
"""
Give liora a translator that turns every card
into an address on a map of meaning.
"""
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name):
    """
    Load a HuggingFace sentence-embedding model
    that turns text into vectors.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name
    )

    return embedding_model
