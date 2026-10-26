#!/usr/bin/env python3
"""
Make Liora speak with Mistral words.
"""


def get_response(llm, prompt):
    """
    Returns response of llm.
    """
    return llm.invoke(input=prompt)
