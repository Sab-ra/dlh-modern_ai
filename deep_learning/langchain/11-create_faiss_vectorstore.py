#!/usr/bin/env python3
"""
Put Liora's cards into a FAISS (Facebook
AI Similarity Search) filing cabinet and
train a retriever to fetch the closest ones.
"""
from langchain_community import vectorstores


def create_faiss_vectorstore(
        docs,
        embedding_model,
        k,
        search_type='similarity'
):
    """
    Embed docs into a FAISS vector store and return it
    together with a retriever that brings back k docs.
    """
    vectorstore = vectorstores.FAISS.from_documents(
        documents=docs,
        embedding=embedding_model
    )
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )

    return vectorstore, retriever
