#!/usr/bin/env python3
"""
Set the rules for Liora, then
line up the users questions.
"""
from langchain_core import messages


def setup_message_sequence(
        system_msg='answer in one short sentence',
        human_msgs=None
):
    """
    Build a list of prompts:
    precieding system message that prevents
    user from tricking model response,
    followed by human messages.
    The system message covers model behaviour
    for all the followers put together.
    """
    if human_msgs is None:
        human_msgs = []
        prompt = [
            messages.SystemMessage(system_msg)
        ]
    for m in human_msgs:
        prompt.append(
            messages.HumanMessage(m)
        )

    return prompt
