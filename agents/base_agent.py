"""
base_agent.py
Purpose: Defines the abstract base class for all trading agents in GoldenSignalsAI. Provides a common interface for agent implementations with the process_signal method.
"""

from typing import Dict, Any
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Abstract base class for all trading agents."""

    @abstractmethod
    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and potentially modify a trading signal.
        Args:
            signal (Dict[str, Any]): Trading signal to process.
        Returns:
            Dict[str, Any]: Modified or original signal.
        """

class RiskAverseAgent(BaseAgent):
    """Agent that modifies signals to reduce risk."""

    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify signal to reduce risk.
        Args:
            signal (Dict[str, Any]): Trading signal to process.
        Returns:
            Dict[str, Any]: Risk-adjusted signal.
        """
        signal['risk_adjusted'] = True
        return signal

class AggressiveAgent(BaseAgent):
    """Agent that modifies signals to maximize potential returns."""

    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify signal to maximize potential returns.
        Args:
            signal (Dict[str, Any]): Trading signal to process.
        Returns:
            Dict[str, Any]: Aggressively modified signal.
        """
        signal['aggressive_mode'] = True
        return signal
