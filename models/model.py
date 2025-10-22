from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract base class for language models with tools.
    """

    def __init__(self):
        """
        Initialize the Model instance.
        The actual model initialization should be done in child classes.
        """
        self.model = None

    @abstractmethod
    def initialize_model(self):
        """
        Initialize the specific model implementation.
        This method must be implemented by child classes.
        """
        pass

    def bind_tools(self, tools):
        """
        Bind the tools to the model.

        Args:
            tools (list): A list of tools to bind to the model.

        Returns:
            The model with the tools bound.
        """
        if self.model is None:
            self.initialize_model()
        return self.model.bind_tools(tools)

    def get_model(self):
        """
        Get the initialized model instance.

        Returns:
            The model instance.
        """
        if self.model is None:
            self.initialize_model()
        return self.model
