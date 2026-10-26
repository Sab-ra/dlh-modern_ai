#!/usr/bin/env python3
"""
Get Mistral AI chat ready for use,
or Liora's other type of voice.
"""
import langchain_mistralai


def load_mistral(model_name, temperature):
    """
    Fetch mistral chat by name, with predefined
    commitment to predictability (temperature).
    """
    llm = langchain_mistralai.ChatMistralAI(
        model_name=model_name,
        temperature=temperature
    )

    return llm
