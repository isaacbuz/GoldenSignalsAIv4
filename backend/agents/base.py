from abc import ABC, abstractmethod
from typing import Type, List

class BaseSignalAgent(ABC):
    """
    Abstract base class for all signal agents.
    All agents inheriting from this class are auto-registered for dynamic discovery.
    """
    registry: List[Type['BaseSignalAgent']] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls not in BaseSignalAgent.registry:
            BaseSignalAgent.registry.append(cls)

    def __init__(self, symbol: str):
        self.symbol = symbol

    @abstractmethod
    def run(self) -> dict:
        """Run the agent and return its signal as a dictionary."""
        pass

    def explain(self) -> str:
        """Return a human-readable explanation of the agent's output."""
        return f"Agent {self.__class__.__name__} does not implement an explanation."
