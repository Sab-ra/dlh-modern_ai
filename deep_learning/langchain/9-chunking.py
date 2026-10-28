#!/usr/bin/env python3
"""
Cut Liora's library into overlapping index cards, so
each answer can use just the right pieces.
"""
from langchain import text_splitter


def split_into_chunks(docs, chunk_size, chunk_overlap):
    """
    Split each Document into chunks of at most chunk_size
    characters, each repeating chunk_overlap characters
    from the previous one.
    """
    splitter = text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(documents=docs)

    return split_docs
