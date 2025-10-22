"""
Main entry point for the LangGraph Weather Agent.

This module serves as the main entry point for the weather and travel assistant.
The application has been refactored into separate modules for better organization:

- config.py: Configuration and logging setup
- state.py: State management
- agent_logic.py: Agent reasoning and model interaction
- workflow.py: LangGraph workflow definition
- chat.py: Interactive chat interface
- models/: Model implementations (OpenAI, Anthropic)
- tools/: Tool implementations (search, weather)
"""

from chat import chat
import config
from config import logger

if __name__ == "__main__":

    logger.info("Starting LangGraph Weather Agent")
    
    chat()
