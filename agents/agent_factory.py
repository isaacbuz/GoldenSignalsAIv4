"""Agent factory for creating agents."""

from typing import Dict, Any, Optional

class AgentFactory:
    """Factory for creating trading agents."""
    
    def __init__(self):
        self.agents = {}
    
    def create_agent(self, agent_type: str, config: Dict[str, Any]):
        """Create an agent of the specified type."""
        # Mock implementation
        return {"type": agent_type, "config": config}
    
    def register_agent(self, agent_type: str, agent_class):
        """Register an agent class."""
        self.agents[agent_type] = agent_class

def get_agent_factory():
    """Get the singleton agent factory."""
    return AgentFactory()
