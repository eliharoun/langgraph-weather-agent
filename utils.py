"""
Utility functions for the LangGraph Weather Agent.
"""

from config import logger

class GraphUtils:
    """Utility class for workflow graph operations."""
    
    @staticmethod
    def save_workflow_graph(workflow_app, filename="workflow_graph.png"):
        """
        Generate and save the workflow graph as a PNG file.
        
        Args:
            workflow_app: Compiled LangGraph workflow application
            filename: Name of the output PNG file
        """
        try:
            # Generate the mermaid PNG data
            png_data = workflow_app.get_graph().draw_mermaid_png()
            
            # Save to file
            with open(filename, "wb") as f:
                f.write(png_data)
            
            logger.info(f"Workflow graph saved as '{filename}'")
            
        except Exception as e:
            logger.warning(f"Could not save workflow graph: {e}")
