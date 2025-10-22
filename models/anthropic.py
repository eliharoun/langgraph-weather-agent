from langchain_anthropic import ChatAnthropic
from .model import Model

class Anthropic(Model):
    """
    Anthropic model implementation that inherits from the base Model class.
    """

    def __init__(self, model_name="claude-3-5-haiku-20241022", temperature=0):
        """
        Initialize the Anthropic Model instance.

        Args:
            model_name (str): The name of the Anthropic model to use. Defaults to "claude-3-5-haiku-20241022".
            temperature (float): The temperature value for the model. Defaults to 0.
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature

    def initialize_model(self):
        """
        Initialize the ChatAnthropic model with the specified parameters.
        """
        self.model = ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature
        )
