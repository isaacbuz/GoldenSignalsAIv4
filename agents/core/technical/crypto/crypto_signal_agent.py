"""
Specialized agent for cryptocurrency trading signals.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ....base import BaseAgent


class CryptoSignalAgent(BaseAgent):
    """Agent specialized for cryptocurrency trading signals with volatility-adjusted parameters."""

    def __init__(
        self,
        name: str = "CryptoSignal",
        ema_short: int = 9,  # Shorter periods for crypto's higher volatility
        ema_long: int = 21,
        volume_ma_period: int = 24,  # Volume analysis for pump detection
        volatility_window: int = 24,
        volatility_threshold: float = 2.0
    ):
        """
        Initialize crypto signal agent.

        Args:
            name: Agent name
            ema_short: Short EMA period
            ema_long: Long EMA period
            volume_ma_period: Volume moving average period
            volatility_window: Window for volatility calculation
            volatility_threshold: Threshold for high volatility detection
        """
        super().__init__(name=name, agent_type="crypto")
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.volume_ma_period = volume_ma_period
        self.volatility_window = volatility_window
        self.volatility_threshold = volatility_threshold

    def calculate_volatility(self, prices: pd.Series) -> float:
        """Calculate current volatility level."""
        returns = prices.pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        return float(volatility.iloc[-1]) if not volatility.empty else 0.0

    def detect_volume_pump(self, volume: pd.Series) -> bool:
        """Detect potential pump and dump based on volume."""
        if len(volume) < self.volume_ma_period:
            return False
        volume_ma = volume.rolling(window=self.volume_ma_period).mean()
        current_ratio = volume.iloc[-1] / volume_ma.iloc[-1]
        return current_ratio > 3.0  # Volume spike threshold

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate crypto-specific signals."""
        try:
            if "close" not in data or "volume" not in data:
                raise ValueError("Missing required price or volume data")

            prices = pd.Series(data["close"])
            volume = pd.Series(data["volume"])

            # Calculate EMAs
            ema_short = prices.ewm(span=self.ema_short, adjust=False).mean()
            ema_long = prices.ewm(span=self.ema_long, adjust=False).mean()

            # Calculate additional indicators
            volatility = self.calculate_volatility(prices)
            is_volume_pump = self.detect_volume_pump(volume)

            # Generate signal
            if ema_short.iloc[-1] > ema_long.iloc[-1]:
                base_action = "buy"
                # Reduce position size in high volatility
                confidence = 0.8 if volatility > self.volatility_threshold else 1.0
                # Reduce confidence if potential pump detected
                if is_volume_pump:
                    confidence *= 0.5
            elif ema_short.iloc[-1] < ema_long.iloc[-1]:
                base_action = "sell"
                confidence = 0.8 if volatility > self.volatility_threshold else 1.0
            else:
                base_action = "hold"
                confidence = 0.0

            return {
                "action": base_action,
                "confidence": confidence,
                "metadata": {
                    "volatility": volatility,
                    "is_volume_pump": is_volume_pump,
                    "ema_short": float(ema_short.iloc[-1]),
                    "ema_long": float(ema_long.iloc[-1])
                }
            }

        except Exception as e:
            self.logger.error(f"Crypto signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
