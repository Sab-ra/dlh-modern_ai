#!/usr/bin/env python3
"""
Carry every help article into Liora's
library, one Document per file.
"""
from langchain_community.document_loaders import(
    DirectoryLoader,
    TextLoader
)


def load_documents(folder_path, file_pattern):
    """
    Find all files in folder_path that match
    file_pattern and read each one with TextLoader
    into a Document.
    """
    loader = DirectoryLoader(
        path=folder_path,
        glob=file_pattern,
        loader_cls=TextLoader
    )
    docs = loader.load()

    return docs
