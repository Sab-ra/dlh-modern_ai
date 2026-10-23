#!/usr/bin/env python3
"""
Give voice to a little silver fox
named Liora.
"""
from langchain_community import llms


def initialize_hf_llm(
        model_name, max_tokens
):
    """
    Set up a HggF text-to-text model
    in LangChain
    """
    llm = llms.HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task='text2text-generation',
        pipeline_kwargs={
            "max_new_tokens": max_tokens,
        }
    )

    return llm
