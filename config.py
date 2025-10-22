"""
Configuration module for the LangGraph Weather Agent.
"""

import logging
from dotenv import load_dotenv, dotenv_values

# Load environment variables
load_dotenv()

dotenv_values = dotenv_values()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
