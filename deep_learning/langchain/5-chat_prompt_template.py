#!/usr/bin/env python3
"""
Hand Liora a two-role script: the director's note
and the consultant's line with blanks to fill in.
"""
from langchain import prompts


def create_chat_prompt_template(
        system_message,
        human_message
):
    """
    Build a ChatPromptTemplate from a system message
    and a human message that holds {placeholders}.
    """
    chat_template = (
        prompts.ChatPromptTemplate.from_messages(
            [
                ('system', system_message),
                ('human', human_message)
            ]
        )
    )

    return chat_template
