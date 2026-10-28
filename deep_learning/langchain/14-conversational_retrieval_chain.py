#!/usr/bin/env python3
"""
Let the smart dog and the reader play together:
one chain that fetches the right cards and answers.
"""
from langchain import chains


def create_conversational_retrieval_chain(
        history_aware_retriever,
        document_chain
):
    """
    Join the history-aware retriever and the document
    chain into one retrieval chain that returns a dict
    with the context and the answer.
    """
    qa_chain = chains.create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=document_chain
    )

    return qa_chain
