from langchain_openai import ChatOpenAI
from .model import Model

class OpenAI(Model):
    """
    OpenAI model implementation that inherits from the base Model class.
    """

    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        """
        Initialize the OpenAI Model instance.

        Args:
            model_name (str): The name of the OpenAI model to use. Defaults to "gpt-4o-mini".
            temperature (float): The temperature value for the model. Defaults to 0.
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature

    def initialize_model(self):
        """
        Initialize the ChatOpenAI model with the specified parameters.
        """
        self.model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature
        )
