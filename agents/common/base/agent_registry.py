from typing import Dict, Optional, Type

from agents.common.base.base_agent import BaseAgent


class AgentRegistry:
    """Registry for managing agent classes."""

    def __init__(self):
        """Initialize the registry."""
        self._agents: Dict[str, Type[BaseAgent]] = {}

    def register_agent(self, name: str) -> callable:
        """Decorator to register an agent class.

        Args:
            name: Unique identifier for the agent

        Returns:
            callable: Decorator function
        """
        def decorator(agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
            if name in self._agents:
                raise ValueError(f"Agent '{name}' is already registered")
            self._agents[name] = agent_class
            return agent_class
        return decorator

    def get_agent(self, name: str) -> Optional[Type[BaseAgent]]:
        """Get an agent class by name.

        Args:
            name: Agent identifier

        Returns:
            Optional[Type[BaseAgent]]: Agent class if found, None otherwise
        """
        return self._agents.get(name)

    def list_agents(self) -> Dict[str, Type[BaseAgent]]:
        """Get all registered agents.

        Returns:
            Dict[str, Type[BaseAgent]]: Dictionary of agent names and classes
        """
        return self._agents.copy()

    def unregister_agent(self, name: str) -> bool:
        """Unregister an agent class.

        Args:
            name: Agent identifier

        Returns:
            bool: True if agent was unregistered, False if not found
        """
        if name in self._agents:
            del self._agents[name]
            return True
        return False

# Global registry instance
registry = AgentRegistry()
