"""
State management module for the LangGraph Weather Agent.
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    """
    The state of our agent - just a list of messages
    
    The 'messages' key will hold the list of messages in the conversation.
    The `add_messages` function tells the graph how to update this key:
    it appends new messages to the list, rather than overwriting it.
    """
    messages: Annotated[list, add_messages]
