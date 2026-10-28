#!/usr/bin/env python3
"""
Give Liora a diary per visitor, so she remembers
the whole conversation on her own.
"""
from langchain_core import runnables


def create_session_aware_chat(
        retrieval_chain,
        get_session_history,
        input_key,
        history_key,
        output_key
):
    """
    Wrap the retrieval chain so each session history
    is loaded before and saved after every answer.
    """
    session_aware_qa_chain = (
        runnables.RunnableWithMessageHistory(
            runnable=retrieval_chain,
            get_session_history=get_session_history,
            input_messages_key=input_key,
            history_messages_key=history_key,
            output_messages_key=output_key
        )
    )

    return session_aware_qa_chain
