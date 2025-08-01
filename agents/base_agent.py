"""
Base Agent Class
Foundation for all trading agents
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all trading agents"""

    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.config = {}
        self.state = {}
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market data and generate signals"""
        pass

    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent with configuration"""
        pass

    def update_state(self, key: str, value: Any):
        """Update agent state"""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get agent state value"""
        return self.state.get(key, default)
