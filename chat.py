"""
Chat interface module for the LangGraph Weather Agent.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from workflow import create_workflow
from utils import GraphUtils
from config import logger

class ChatInterface:
    """Chat interface for the weather and travel assistant."""
    
    def __init__(self):
        """Initialize the chat interface with the workflow."""
        self.app = create_workflow()
        
        # Generate and save workflow graph as PNG
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
        
        # Start with a system message to set behavior
        messages = [self.system_message]
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue
            
            # Add user message
            messages.append(HumanMessage(content=user_input))
            
            # Run the agent
            print("\nAssistant: ", end="", flush=True)
            
            try:
                # To simulate a continuous conversation, we need a thread ID.
                # The checkpointer uses this ID to store and retrieve the correct conversation state.
                # For a simple command-line app, we can just use a fixed ID.
                config = {"configurable": {"thread_id": "1"}}

                result = self.app.invoke({"messages": messages}, config=config)
                
                # Get the final response
                final_message = result["messages"][-1]
                print(final_message.content)
                print()
                
                # Update messages for next turn
                messages = result["messages"]
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                print(f"Sorry, I encountered an error: {e}")
                print()

def chat():
    """Function to start the chat interface."""
    interface = ChatInterface()
    interface.run()
