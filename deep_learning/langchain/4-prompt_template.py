#!/usr/bin/env python3
"""
Give Liora a reusable script with blanks to fill in.
"""
from langchain_core import prompts


def create_prompt_template(
        template_str,
        input_variables
):
    """
    Build a PromptTemplate from a string with
    {placeholders} and the list of their names.
    """
    template = prompts.PromptTemplate(
        template=template_str,
        input_variables=input_variables
    )

    return template
