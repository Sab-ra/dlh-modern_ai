#!/usr/bin/env python3
"""
Let Liora read the fetched cards, remember the talk
so far, and answer the new question from them.
"""
from langchain_core import prompts
from langchain.chains import combine_documents


def document_answer_chain(
        llm, system_message, human_message
):
    """
    Build a chat prompt (system, chat_history, human)
    and a document chain that stuffs the retrieved docs
    into {context} and asks the llm.
    """
    prompt_get_answer = (
        prompts.ChatPromptTemplate.from_messages([
            ("system", system_message),
            prompts.MessagesPlaceholder(
                variable_name='chat_history'
            ),
            ("human", human_message)
        ])
    )
    qa_document_chain = combine_documents.create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_get_answer
    )

    return prompt_get_answer, qa_document_chain
