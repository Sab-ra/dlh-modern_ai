#!/usr/bin/env python3
"""
Let Liora listen to the whole conversation and
rewrite a vague question before the dog fetches.
"""
from langchain_core import prompts
from langchain import chains


def build_history_aware_retriever(
        llm, retriever, human_message
):
    """
    Build a prompt (chat_history, then human input) and
    a retriever that first lets the llm turn the talk
    into a clear search query, then fetches documents.
    """
    prompt_search_query = (
        prompts.ChatPromptTemplate.from_messages(
            [
                prompts.MessagesPlaceholder(
                    variable_name='chat_history'
                ),
                ("human", human_message)
            ]
        )
    )
    history_aware_retriever = (
        chains.create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=prompt_search_query
        )
    )

    return prompt_search_query, history_aware_retriever
