from abc import ABC, abstractmethod

class BaseTool(ABC):
    """
    Abstract base class for all agent tools.
    All tools must inherit from this class and implement the .use() method.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def use(self, query: str) -> str:
        """
        The main method for the tool.
        Takes a query as input and returns a string with the result.
        """
        pass