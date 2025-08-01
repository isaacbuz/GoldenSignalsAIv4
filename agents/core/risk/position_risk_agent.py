"""
Position risk management agent.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class PositionRiskAgent(BaseAgent):
    """Agent that manages position risk and generates risk-adjusted signals."""

    def __init__(
        self,
        name: str = "PositionRisk",
        max_position_size: float = 0.2,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.05,
        max_drawdown: float = 0.1,
        volatility_window: int = 20
    ):
        """
        Initialize position risk agent.

        Args:
            name: Agent name
            max_position_size: Maximum position size as fraction of portfolio
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_drawdown: Maximum allowed drawdown
            volatility_window: Window for volatility calculation
        """
        super().__init__(name=name, agent_type="risk")
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown = max_drawdown
        self.volatility_window = volatility_window

    def calculate_position_size(
        self,
        prices: pd.Series,
        portfolio_value: float,
        signal_confidence: float
    ) -> float:
        """Calculate risk-adjusted position size."""
        try:
            # Calculate volatility
            returns = prices.pct_change()
            volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]

            # Adjust position size based on volatility and confidence
            vol_factor = 1.0 / (1.0 + volatility)
            position_size = self.max_position_size * signal_confidence * vol_factor

            # Calculate maximum position value
            max_position_value = portfolio_value * position_size

            # Calculate number of shares
            current_price = prices.iloc[-1]
            shares = int(max_position_value / current_price)

            return shares

        except Exception as e:
            logger.error(f"Position size calculation failed: {str(e)}")
            return 0

    def calculate_risk_metrics(
        self,
        prices: pd.Series,
        position: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate current risk metrics for position."""
        try:
            if not position or "entry_price" not in position:
                return {}

            current_price = prices.iloc[-1]
            entry_price = position["entry_price"]

            # Calculate returns and drawdown
            returns = (current_price / entry_price - 1)
            max_price = prices.max()
            drawdown = (max_price - current_price) / max_price

            # Calculate risk metrics
            stop_loss_hit = returns <= -self.stop_loss_pct
            take_profit_hit = returns >= self.take_profit_pct
            max_drawdown_hit = drawdown >= self.max_drawdown

            return {
                "returns": float(returns),
                "drawdown": float(drawdown),
                "stop_loss_hit": stop_loss_hit,
                "take_profit_hit": take_profit_hit,
                "max_drawdown_hit": max_drawdown_hit
            }

        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {str(e)}")
            return {}

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and current position for risk management."""
        try:
            if "close_prices" not in data or "portfolio_value" not in data:
                raise ValueError("Missing required data for risk management")

            prices = pd.Series(data["close_prices"])
            portfolio_value = float(data["portfolio_value"])
            position = data.get("current_position", {})
            signal_confidence = data.get("signal_confidence", 0.5)

            # Calculate position size for new trades
            position_size = self.calculate_position_size(
                prices,
                portfolio_value,
                signal_confidence
            )

            # Calculate risk metrics for existing position
            risk_metrics = self.calculate_risk_metrics(prices, position)

            # Generate risk management signal
            if not position:
                action = "hold"
                confidence = 0.0
            elif (
                risk_metrics.get("stop_loss_hit", False) or
                risk_metrics.get("max_drawdown_hit", False)
            ):
                action = "exit"
                confidence = 1.0
            elif risk_metrics.get("take_profit_hit", False):
                action = "reduce"
                confidence = 0.8
            else:
                action = "hold"
                confidence = 0.0

            return {
                "action": action,
                "confidence": confidence,
                "metadata": {
                    "position_size": position_size,
                    "risk_metrics": risk_metrics,
                    "max_position_size": self.max_position_size,
                    "stop_loss_pct": self.stop_loss_pct,
                    "take_profit_pct": self.take_profit_pct,
                    "max_drawdown": self.max_drawdown
                }
            }

        except Exception as e:
            logger.error(f"Position risk management failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
