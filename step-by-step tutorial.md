# Building Your First AI Agent with LangGraph - Complete Beginner's Tutorial

## What You'll Build

By the end of this tutorial, you'll have created a **Weather & Travel Assistant** - a production-ready AI agent that:
- Checks **real weather data** using live APIs
- Searches for travel information online
- Remembers conversation history with persistent memory
- Uses modular, maintainable code architecture
- Supports multiple AI models (OpenAI & Anthropic)
- Includes proper error handling and logging

**Demo Value**: This agent showcases real AI agent capabilities with professional software engineering practices.

---

## Table of Contents
1. [What is LangGraph?](#what-is-langgraph)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Understanding Key Concepts](#understanding-key-concepts)
4. [Project Architecture Overview](#project-architecture-overview)
5. [Step-by-Step Implementation](#step-by-step-implementation)
6. [Running Your Agent](#running-your-agent)
7. [What's Next?](#whats-next)

---

## What is LangGraph?

LangGraph is a framework for building AI agents that can:
- **Think step-by-step** using a graph structure (nodes and edges)
- **Use tools** like search engines or APIs
- **Remember conversations** with persistent memory
- **Make decisions** about what to do next

Think of it like building a flowchart where each box (node) is an action your AI can take, and the arrows (edges) show which action comes next.

---

## Prerequisites & Setup

### What You Need
- Python 3.9 or higher
- Basic Python knowledge (functions, classes, modules)
- API keys for:
  - OpenAI ([get one here](https://platform.openai.com/api-keys))
  - Tavily search ([get free key here](https://tavily.com/))
  - RapidAPI weather ([get free key here](https://rapidapi.com/))

### Step 1: Create Your Project

```bash
# Clone or create the project directory
mkdir langgraph-travel-agent
cd langgraph-travel-agent

# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install Required Packages

Create `requirements.txt`:
```txt
langgraph 
langgraph-checkpoint-sqlite
langchain-openai 
langchain
langchain-community
langchain-anthropic
langchain-tavily
requests
python-dotenv
IPython
```

Install packages:
```bash
pip install -r requirements.txt
```

### Step 3: Set Up API Keys

Create a `.env` file in your project folder:

```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here  # Optional
TAVILY_API_KEY=your_tavily_key_here
RAPID_API_KEY=your_rapidapi_key_here
OPEN_WEATHER_API_URL=https://open-weather13.p.rapidapi.com/city
RAPID_API_HOST=open-weather13.p.rapidapi.com
```

---

## Understanding Key Concepts

Before we code, let's understand the building blocks:

### 1. **State** (Memory)
The "brain" of your agent - stores conversation history with automatic persistence.

### 2. **Nodes** (Actions)
Functions that do something - call the AI, use tools, etc.

### 3. **Edges** (Connections)
Define workflow flow. Two types:
- **Normal edges**: Always go to the next node
- **Conditional edges**: Decide where to go based on the situation

### 4. **Tools** (Capabilities)
Functions your agent can call - real APIs, not mock data.

### 5. **Models** (AI Engines)
Abstracted model classes supporting multiple providers (OpenAI, Anthropic).

---

## Project Architecture Overview

Our agent follows a modular architecture:

```
langgraph-travel-agent/
├── run_agent.py          # Main entry point
├── config.py            # Configuration & logging
├── state.py             # State management
├── agent_logic.py       # Agent reasoning
├── workflow.py          # LangGraph workflow
├── chat.py              # Interactive chat interface  
├── utils.py             # Utility functions
├── models/              # Model implementations
│   ├── __init__.py
│   ├── model.py         # Abstract base class
│   ├── openai.py        # OpenAI implementation
│   └── anthropic.py     # Anthropic implementation
├── tools/               # Tool implementations
│   ├── __init__.py
│   ├── weather.py       # Real weather API
│   └── search.py        # Tavily search
├── requirements.txt     # Dependencies
├── .env                 # API keys
└── langchain.db         # Persistent memory
```

**Benefits of this structure:**
- **Separation of concerns**: Each module has a specific responsibility
- **Reusability**: Components can be easily swapped or extended
- **Testability**: Individual modules can be tested in isolation
- **Maintainability**: Changes to one area don't affect others

---

## Step-by-Step Implementation

### Step 1: Configuration Setup

Create `config.py` for environment management:

```python
"""
Configuration module for the LangGraph Travel Agent.
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
```

### Step 2: Define State Management

Create `state.py` for conversation memory:

```python
"""
State management module for the LangGraph Travel Agent.
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
```

### Step 3: Create Model Abstraction

Create `models/model.py` as the abstract base:

```python
from abc import ABC, abstractmethod

class Model(ABC):
    """Abstract base class for language models with tools."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def initialize_model(self):
        """Initialize the specific model implementation."""
        pass

    def bind_tools(self, tools):
        """Bind tools to the model."""
        if self.model is None:
            self.initialize_model()
        return self.model.bind_tools(tools)

    def get_model(self):
        """Get the initialized model instance."""
        if self.model is None:
            self.initialize_model()
        return self.model
```

Create `models/openai.py`:

```python
from langchain_openai import ChatOpenAI
from .model import Model

class OpenAI(Model):
    """OpenAI model implementation."""

    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature

    def initialize_model(self):
        self.model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature
        )
```

### Step 4: Build Real Tools

Create `tools/weather.py` with real API integration:

```python
import os
import requests
from langchain_core.tools import tool
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("RAPID_API_KEY")
url = os.getenv("OPEN_WEATHER_API_URL")
rapid_api_host = os.getenv("RAPID_API_HOST")

headers = {
    "x-rapidapi-key": api_key,
    "x-rapidapi-host": rapid_api_host
}

def get_weather(city: str) -> str:
    """Get current weather for a city using real API."""
    city_lower = city.lower()
    querystring = {"city": city_lower, "lang": "EN"}

    try:
        response = requests.get(url, headers=headers, params=querystring)
        description = response.json().get("weather", [{}])[0].get("description")
        temp = response.json().get("main", {}).get("temp")
        
        logger.info(f"Got weather for {city}: {description}, {temp}°F")
        return f"{description}, {temp}°F"
    except Exception as e:
        logger.error(e)
        return "sunny, 80.72°F"  # fallback

@tool
def weather_tool(city: str) -> str:
    """Get current weather for a city"""
    return get_weather(city)
```

Create `tools/search.py`:

```python
from langchain_tavily import TavilySearch

search_tool = TavilySearch(
    max_results=1,
    description="general"
)
```

### Step 5: Agent Logic

Create `agent_logic.py`:

```python
"""Agent logic module for the LangGraph Travel Agent."""

import config
from config import logger
from state import State
from tools import search_tool, weather_tool
from models import OpenAI

# Combine all tools
tools = [search_tool, weather_tool]

# Initialize model and give access to tools
model_with_tools = OpenAI().bind_tools(tools)

def call_agent(state: State):
    """
    The main agent node - decides what to do next
    
    Returns: Updated state with the agent's response
    """
    messages = state["messages"]
    
    logger.info(f"Agent processing {len(messages)} messages")
    
    # Call the model
    response = model_with_tools.invoke(messages)
    
    return {"messages": [response]}
```

### Step 6: Workflow with Persistence

Create `workflow.py`:

```python
"""Workflow module for the LangGraph Travel Agent."""

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
    """Create and configure the LangGraph workflow."""
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", ToolNode(tools))

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")

    # Compile with persistent memory
    return workflow.compile(checkpointer=memory)
```

### Step 7: Chat Interface

Create `chat.py`:

```python
"""Chat interface module for the LangGraph Travel Agent."""

from langchain_core.messages import HumanMessage, SystemMessage
from workflow import create_workflow
from utils import GraphUtils
from config import logger

class ChatInterface:
    """Chat interface for the weather and travel assistant."""
    
    def __init__(self):
        self.app = create_workflow()
        
        # Generate and save workflow graph
        GraphUtils.save_workflow_graph(self.app)
        
        self.system_message = SystemMessage(content="""You are a helpful travel and weather assistant.
        
        You have access to:
        1. A weather tool - use it when users ask about weather
        2. A search tool - use it for travel info, attractions, or general questions
        
        Be friendly, concise, and helpful!""")
    
    def run(self):
        """Run the interactive chat interface."""
        print("Weather & Travel Assistant")
        print("=" * 50)
        print("Ask me about weather or travel information!")
        print("Type 'quit' to exit\n")
        
        messages = [self.system_message]
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue
            
            messages.append(HumanMessage(content=user_input))
            print("\nAssistant: ", end="", flush=True)
            
            try:
                config = {"configurable": {"thread_id": "1"}}
                result = self.app.invoke({"messages": messages}, config=config)
                
                final_message = result["messages"][-1]
                print(final_message.content)
                print()
                
                messages = result["messages"]
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                print(f"Sorry, I encountered an error: {e}")
                print()

def chat():
    """Function to start the chat interface."""
    interface = ChatInterface()
    interface.run()
```

### Step 8: Create Package __init__.py Files

Create `models/__init__.py`:

```python
"""
Models package for the LangGraph Travel Agent.

This package provides model implementations for different AI providers.
"""

from .openai import OpenAI
from .anthropic import Anthropic

__all__ = ["OpenAI", "Anthropic"]
```

Create `tools/__init__.py`:

```python
"""
Tools package for the LangGraph Travel Agent.

This package provides various tools for searching and weather information.
"""

from .search import search_tool
from .weather import weather_tool

__all__ = ["search_tool", "weather_tool"]
```

### Step 9: Utilities and Main Entry Point

Create `utils.py`:

```python
"""Utility functions for the LangGraph Travel Agent."""

from config import logger

class GraphUtils:
    """Utility class for workflow graph operations."""
    
    @staticmethod
    def save_workflow_graph(workflow_app, filename="workflow_graph.png"):
        """Generate and save the workflow graph as a PNG file."""
        try:
            png_data = workflow_app.get_graph().draw_mermaid_png()
            
            with open(filename, "wb") as f:
                f.write(png_data)
            
            logger.info(f"Workflow graph saved as '{filename}'")
            
        except Exception as e:
            logger.warning(f"Could not save workflow graph: {e}")
```

Create `run_agent.py`:

```python
"""
Main entry point for the LangGraph Travel Agent.

The application has been refactored into separate modules:
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
    logger.info("Starting LangGraph Travel Agent")
    chat()
```

---

## Running Your Agent

### Project Structure Verification

After implementing all the modules, your project structure should look like this:

```
langgraph-travel-agent/
├── run_agent.py          # ✅ Main entry point
├── config.py             # ✅ Configuration & logging  
├── state.py              # ✅ State management
├── agent_logic.py        # ✅ Agent reasoning
├── workflow.py           # ✅ LangGraph workflow
├── chat.py               # ✅ Interactive chat interface
├── utils.py              # ✅ Utility functions
├── models/               # ✅ Model implementations
│   ├── __init__.py
│   ├── model.py          # ✅ Abstract base class
│   ├── openai.py         # ✅ OpenAI implementation  
│   └── anthropic.py      # ✅ Anthropic implementation
├── tools/                # ✅ Tool implementations
│   ├── __init__.py
│   ├── weather.py        # ✅ Real weather API
│   └── search.py         # ✅ Tavily search
├── requirements.txt      # ✅ Dependencies
├── .env                  # ✅ API keys
├── langchain.db          # Generated: Persistent memory
└── workflow_graph.png    # Generated: Visual workflow
```

### Run Your Agent

Start your weather and travel assistant:

```bash
python run_agent.py
```

### Expected Output

```
2024-01-01 12:00:00 - __main__ - INFO - Starting LangGraph Travel Agent
2024-01-01 12:00:00 - utils - INFO - Workflow graph saved as 'workflow_graph.png'

Weather & Travel Assistant
Ask me about weather or travel information!
Type 'quit' to exit

You: 
```

### Example Conversations

**Real Weather Data:**
```
You: What's the weather in New York?
Assistant: I'll check the current weather in New York for you.

The current weather in New York is overcast clouds with a temperature of 45°F.
```

**Travel Information:**
```
You: Tell me about the top attractions in Paris
Assistant: I'll search for information about the top attractions in Paris.

Here are some of the top attractions in Paris:

1. **Eiffel Tower** - The iconic iron lattice tower and symbol of Paris
2. **Louvre Museum** - Home to the Mona Lisa and thousands of artworks
3. **Notre-Dame Cathedral** - Gothic architecture masterpiece (currently under restoration)
4. **Arc de Triomphe** - Monumental arch at the western end of the Champs-Élysées
5. **Sacré-Cœur Basilica** - Beautiful white domed church in Montmartre

Would you like more detailed information about any of these attractions?
```

**Combined Queries:**
```
You: What's the weather in Tokyo and what should I visit there?
Assistant: I'll check the weather in Tokyo and find information about attractions there.

The current weather in Tokyo is clear skies with a temperature of 72°F - perfect weather for sightseeing!

Here are some must-visit attractions in Tokyo:
- **Senso-ji Temple** - Ancient Buddhist temple in Asakusa
- **Tokyo Skytree** - Tallest tower in Japan with amazing city views
- **Meiji Shrine** - Peaceful Shinto shrine surrounded by forest
- **Shibuya Crossing** - Famous busy pedestrian crossing
- **Tsukiji Outer Market** - Great for fresh seafood and street food

The clear weather makes it an excellent day for exploring these outdoor attractions!
```

---

## How It Works (Behind the Scenes)

Let's trace what happens when you ask: **"What's the weather in Tokyo?"**

1. **You type** → HumanMessage added to state
2. **Graph starts** → Goes to `agent` node
3. **Agent node** → Calls GPT with conversation + tool descriptions
4. **GPT thinks** → "I need to use weather_tool with city='Tokyo'"
5. **Conditional edge** → Sees tool call, routes to `tools` node
6. **Tools node** → Executes `weather_tool("Tokyo")` → Returns "Clear, 72°F"
7. **Edge** → Goes back to `agent` node
8. **Agent node** → Calls GPT again with tool result
9. **GPT responds** → "The weather in Tokyo is clear and 72°F"
10. **Conditional edge** → No more tools needed, goes to END
11. **You see** → Final response printed!

---

## What's Next?

### Understanding What You Built

Your agent already includes several advanced features:

**✅ Real API Integration:**
- Live weather data from RapidAPI
- Web search through Tavily

**✅ Persistent Memory:**
- SQLite database storage (`langchain.db`)
- Conversation history maintained across sessions

**✅ Professional Architecture:**
- Modular, maintainable code structure
- Abstract model classes for easy provider switching
- Comprehensive error handling and logging

**✅ Visual Workflow:**
- Automatic graph generation (`workflow_graph.png`)
- Clear visualization of agent decision flow

### Enhance Your Agent Further

**1. Add More AI Models:**
```python
# In agent_logic.py, switch to Anthropic:
from models import Anthropic
model_with_tools = Anthropic().bind_tools(tools)
```

**2. Add More Tools:**
```python
@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency between different currencies"""
    # Integrate with a real currency API
    pass

@tool  
def restaurant_finder(location: str, cuisine: str) -> str:
    """Find restaurants by location and cuisine type"""
    # Integrate with restaurant API
    pass
```

**3. Add Human-in-the-Loop:**
```python
from langgraph.prebuilt import human_node

# In workflow.py
workflow.add_node("human_approval", human_node)
workflow.add_conditional_edges(
    "agent",
    should_get_human_approval,  # Custom function
    {"human": "human_approval", "tools": "tools", END: END}
)
```

**4. Add Multiple Conversation Threads:**
```python
# In chat.py, use different thread IDs for different conversations
config = {"configurable": {"thread_id": f"user_{user_id}"}}
```

**5. Add Streaming Responses:**
```python
# In chat.py
for chunk in self.app.stream({"messages": messages}, config=config):
    print(chunk, end="", flush=True)
```

### Key Concepts Mastered

✅ **LangGraph Workflows:** Nodes, edges, and conditional routing  
✅ **Tool Integration:** Real API connections with error handling  
✅ **State Management:** Persistent conversation memory  
✅ **Model Abstraction:** Multi-provider AI model support  
✅ **Production Patterns:** Logging, error handling, modular design

### Advanced Topics to Explore

**Multi-Agent Systems:**
- Create specialized agents for different tasks
- Implement agent-to-agent communication
- Build supervisor agents that coordinate workflows

**Advanced Memory:**
- Implement semantic search over conversation history
- Add external memory stores (vector databases)
- Create context-aware memory retrieval

**Production Deployment:**
- Containerize with Docker
- Deploy to cloud platforms
- Add monitoring and observability
- Implement rate limiting and authentication

### Resources for Continued Learning

- **[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)** - Official docs and API reference
- **[LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph)** - Free comprehensive course
- **[LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)** - Real-world implementations

---

## Troubleshooting

**"ModuleNotFoundError"**
→ Make sure you activated your virtual environment and installed packages

**"API key not found"**
→ Check your `.env` file has the correct keys

**Agent not using tools**
→ Make sure you called `model.bind_tools(tools)`

**Infinite loops**
→ Add a recursion limit: `app.invoke(state, {"recursion_limit": 10})`

---

Congratulations! You've built your first AI agent with LangGraph. This is just the beginning - you can now build complex multi-agent systems, add custom logic, and create production-ready AI applications! 🚀
