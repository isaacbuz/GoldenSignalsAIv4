from typing import Dict, Any
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    @abstractmethod
    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Process and potentially modify a trading signal."""

class RiskAverseAgent(BaseAgent):
    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Modify signal to reduce risk."""
        signal['risk_adjusted'] = True
        return signal

class AggressiveAgent(BaseAgent):
    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Modify signal to maximize potential returns."""
        signal['aggressive_mode'] = True
        return signal
