"""Mock database module for testing"""
from datetime import datetime
from typing import Any, Dict, List, Optional


class DatabaseManager:
    """Mock database manager"""

    def __init__(self, *args, **kwargs):
        self.signals = []
        self.agent_states = {}
        self.performance_data = {}

    async def store_signal(self, signal_data: Dict[str, Any]) -> None:
        self.signals.append(signal_data)

    async def update_agent_performance(self, agent_id: str, data: Dict[str, Any]) -> None:
        self.performance_data[agent_id] = data

    async def save_agent_state(self, agent_id: str, name: str, state: Dict[str, Any]) -> None:
        self.agent_states[agent_id] = state

    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self.agent_states.get(agent_id)

    async def get_market_data(self, symbol: str, since: datetime, limit: int) -> List[Any]:
        return []

    async def get_signals(self, **kwargs) -> List[Any]:
        return []
