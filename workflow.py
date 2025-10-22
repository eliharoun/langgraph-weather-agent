"""
Workflow module for the LangGraph Weather Agent.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from state import State
from agent_logic import call_agent, tools
import sqlite3


# Add memory persistence
conn = sqlite3.connect('langchain.db', check_same_thread=False)

memory = SqliteSaver(conn)

def create_workflow():
    """
    Create and configure the LangGraph workflow.
    
    Returns:
        Compiled workflow graph
    """
    # Create the graph
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("agent", call_agent)  # The thinking node
    workflow.add_node("tools", ToolNode(tools))  # The tool-using node

    # Add edges
    workflow.add_edge(START, "agent")  # Start with the agent

    # Add conditional edge - decides what to do after agent thinks
    workflow.add_conditional_edges(
        "agent",  # From the agent node
        tools_condition,  # Decision function (built-in!)
        {
            "tools": "tools",  # If agent wants tools, go to tools node
            END: END  # If agent has final answer, end
        }
    )

    # After using tools, go back to agent to process results
    workflow.add_edge("tools", "agent")

    # Compile the graph into a runnable agent. Include a checkpointer

    return workflow.compile(checkpointer=memory)
