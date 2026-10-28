#!/usr/bin/env python3
"""
Hand Liora a two-role script: the director's note
and the consultant's line with blanks to fill in.
"""
from langchain import prompts


def create_chat_prompt_template(
        system_message,
        human_message,
        placeholder=None
):
    """
    Build a ChatPromptTemplate from a system message
    and a human message that holds {placeholders}.
    If placeholder is given (e.g. "{agent_scratchpad}"),
    add a slot where the agent keeps its tool notes.
    """
    messages = [
                ('system', system_message),
                ('human', human_message)
    ]
    if placeholder is not None:
        messages.append(('placeholder', placeholder))

    chat_template = prompts.ChatPromptTemplate.from_messages(
        messages
    )

    return chat_template
