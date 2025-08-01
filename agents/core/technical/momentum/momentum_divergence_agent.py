"""
Momentum divergence analysis agent.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ....base import BaseAgent

logger = logging.getLogger(__name__)

class MomentumDivergenceAgent(BaseAgent):
    """Agent that detects price-momentum divergences for trading signals."""

    def __init__(
        self,
        name: str = "MomentumDivergence",
        lookback_period: int = 20,
        divergence_threshold: float = 0.1,
        momentum_window: int = 14
    ):
        """
        Initialize momentum divergence agent.

        Args:
            name: Agent name
            lookback_period: Period for divergence detection
            divergence_threshold: Minimum divergence for signal generation
            momentum_window: Window for momentum calculation
        """
        super().__init__(name=name, agent_type="technical")
        self.lookback_period = lookback_period
        self.divergence_threshold = divergence_threshold
        self.momentum_window = momentum_window

    def calculate_momentum(self, prices: pd.Series) -> pd.Series:
        """Calculate price momentum using ROC (Rate of Change)."""
        try:
            return prices.pct_change(self.momentum_window)
        except Exception as e:
            logger.error(f"Momentum calculation failed: {str(e)}")
            return pd.Series()

    def detect_divergence(
        self,
        prices: pd.Series,
        momentum: pd.Series
    ) -> Dict[str, bool]:
        """Detect bullish and bearish divergences."""
        try:
            if len(prices) < self.lookback_period:
                return {"bullish": False, "bearish": False}

            # Get recent price and momentum extremes
            recent_prices = prices[-self.lookback_period:]
            recent_momentum = momentum[-self.lookback_period:]

            price_high = recent_prices.max()
            price_low = recent_prices.min()
            mom_high = recent_momentum.max()
            mom_low = recent_momentum.min()

            # Calculate percentage changes
            price_change = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1)
            mom_change = (recent_momentum.iloc[-1] / recent_momentum.iloc[0] - 1)

            # Detect divergences
            bullish = (
                price_change < -self.divergence_threshold and
                mom_change > self.divergence_threshold
            )

            bearish = (
                price_change > self.divergence_threshold and
                mom_change < -self.divergence_threshold
            )

            return {
                "bullish": bullish,
                "bearish": bearish
            }

        except Exception as e:
            logger.error(f"Divergence detection failed: {str(e)}")
            return {"bullish": False, "bearish": False}

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate divergence signals."""
        try:
            if "close_prices" not in data:
                raise ValueError("Close prices not found in market data")

            prices = pd.Series(data["close_prices"])
            momentum = self.calculate_momentum(prices)

            if len(momentum) < self.lookback_period:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "error": "Insufficient data for divergence detection"
                    }
                }

            # Detect divergences
            divergences = self.detect_divergence(prices, momentum)

            # Generate signal
            if divergences["bullish"]:
                action = "buy"
                confidence = abs(
                    momentum.iloc[-1] / momentum[-self.lookback_period:].mean()
                )
            elif divergences["bearish"]:
                action = "sell"
                confidence = abs(
                    momentum.iloc[-1] / momentum[-self.lookback_period:].mean()
                )
            else:
                action = "hold"
                confidence = 0.0

            return {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {
                    "momentum": float(momentum.iloc[-1]),
                    "lookback_period": self.lookback_period,
                    "divergence_threshold": self.divergence_threshold,
                    "divergences": divergences
                }
            }

        except Exception as e:
            logger.error(f"Momentum divergence processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
