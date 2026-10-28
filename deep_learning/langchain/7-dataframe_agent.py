#!/usr/bin/env python3
"""
Give Liora a spreadsheet and a Python calculator
so she can count customers instead of guesing.
"""
from langchain_experimental import agents


def create_dataframe_agent(llm, df):
    """
    Build a pandas DataFrame agent that writes and
    runs pandas code on df to answer questions.
    """
    dataframe_agent = agents.create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        allow_dangerous_code=True,
        verbose=True,
        agent_executor_kwargs={
            "handle_parsing_errors": True
        }
    )

    return dataframe_agent
