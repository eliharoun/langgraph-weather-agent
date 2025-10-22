"""
Models package for the LangGraph Weather Agent.

This package provides different model implementations that inherit from the base Model class.
"""

from .openai import OpenAI
from .anthropic import Anthropic

__all__ = ["OpenAI", "Anthropic"]
