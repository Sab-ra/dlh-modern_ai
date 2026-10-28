#!/usr/bin/env python3
"""
Give Liora a toolbox and a notepad, so she can
look things up instead of guessing.
"""
from langchain import agents
from langchain import agent_toolkits


def load_agent_with_toolkit(
        llm,
        toolkit_names,
        prompt
):
    """
    Load the named tools, build a tool-calling agent
    from the llm and prompt, and wrap it in a verbose
    AgentExecutor that runs the think-call-read loop.
    """
    tools = agent_toolkits.load_tools(
        tool_names=toolkit_names
    )
    agent = agents.create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    tool_agent = agents.AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    return tool_agent
