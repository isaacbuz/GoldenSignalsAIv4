"""
Combined RSI and MACD technical analysis agent.
"""
import logging
from typing import Any, Dict, Optional

import pandas as pd

from ....base import BaseAgent
from .macd_agent import MACDAgent
from .rsi_agent import RSIAgent

logger = logging.getLogger(__name__)

class RSIMACDAgent(BaseAgent):
    """Agent that combines RSI and MACD signals for enhanced trading decisions."""

    def __init__(
        self,
        name: str = "RSI-MACD",
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        macd_threshold: float = 0.2
    ):
        """
        Initialize RSI-MACD agent.

        Args:
            name: Agent name
            rsi_period: RSI calculation period
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            macd_threshold: MACD divergence threshold
        """
        super().__init__(name=name, agent_type="technical")

        # Initialize sub-agents
        self.rsi_agent = RSIAgent(
            period=rsi_period,
            overbought=rsi_overbought,
            oversold=rsi_oversold
        )

        self.macd_agent = MACDAgent(
            fast_period=macd_fast,
            slow_period=macd_slow,
            signal_period=macd_signal,
            divergence_threshold=macd_threshold
        )

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data using both RSI and MACD for signal generation."""
        try:
            # Get signals from both indicators
            rsi_signal = self.rsi_agent.process(data)
            macd_signal = self.macd_agent.process(data)

            # Check for errors in either signal
            if "error" in rsi_signal.get("metadata", {}) or "error" in macd_signal.get("metadata", {}):
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "error": "Error in indicator calculations",
                        "rsi_error": rsi_signal.get("metadata", {}).get("error"),
                        "macd_error": macd_signal.get("metadata", {}).get("error")
                    }
                }

            # Generate combined signal
            rsi_action = rsi_signal["action"]
            macd_action = macd_signal["action"]

            # Only generate signal if both indicators agree
            if rsi_action == macd_action and rsi_action != "hold":
                action = rsi_action
                # Use geometric mean of confidences for combined confidence
                confidence = (rsi_signal["confidence"] * macd_signal["confidence"]) ** 0.5
            else:
                action = "hold"
                confidence = 0.0

            return {
                "action": action,
                "confidence": confidence,
                "metadata": {
                    "rsi": {
                        "value": rsi_signal["metadata"].get("rsi"),
                        "action": rsi_action,
                        "confidence": rsi_signal["confidence"]
                    },
                    "macd": {
                        "value": macd_signal["metadata"].get("macd"),
                        "action": macd_action,
                        "confidence": macd_signal["confidence"]
                    },
                    "agreement": rsi_action == macd_action
                }
            }

        except Exception as e:
            logger.error(f"RSI-MACD signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
