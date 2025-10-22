"""
Agent logic module for the LangGraph Weather Agent.
"""

import config
from config import logger
from state import State
from tools import search_tool, weather_tool
from models import OpenAI

# Combine all tools
tools = [search_tool, weather_tool]

# Initialize model and give the model access to tools
model_with_tools = OpenAI().bind_tools(tools)

def call_agent(state: State):
    """
    The main agent node - decides what to do next (thinking)
    
    Args:
        state: Current state containing messages
        
    Returns: 
        Updated state with the agent's response
    """
    messages = state["messages"]
    
    logger.info(f"Agent processing {len(messages)} messages")
    
    # Call the model
    response = model_with_tools.invoke(messages)
    
    # Return the response (it will be added to state automatically)
    return {"messages": [response]}
