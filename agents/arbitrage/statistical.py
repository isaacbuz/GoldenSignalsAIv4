"""
Statistical arbitrage agent implementation.
"""
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ArbitrageOpportunity, BaseArbitrageAgent

logger = logging.getLogger(__name__)

class StatisticalArbitrageAgent(BaseArbitrageAgent):
    """Agent that finds statistical arbitrage opportunities."""

    def __init__(
        self,
        name: str = "StatisticalArbitrage",
        min_spread: float = 0.01,
        min_volume: float = 100.0,
        max_slippage: float = 0.002,
        max_latency_ms: int = 200,
        fee_rate: float = 0.001,
        window: int = 20,
        zscore_threshold: float = 2.0,
        mean_reversion_threshold: float = 0.7,
        data_fetcher: Optional[Callable] = None
    ):
        """
        Initialize statistical arbitrage agent.

        Args:
            name: Agent name
            min_spread: Minimum spread to consider
            min_volume: Minimum volume for opportunities
            max_slippage: Maximum acceptable slippage
            max_latency_ms: Maximum acceptable latency
            fee_rate: Trading fee rate
            window: Rolling window for statistics
            zscore_threshold: Z-score threshold for signals
            mean_reversion_threshold: Required mean reversion strength
            data_fetcher: Function to fetch historical data
        """
        super().__init__(
            name=name,
            min_spread=min_spread,
            min_volume=min_volume,
            max_slippage=max_slippage,
            max_latency_ms=max_latency_ms,
            fee_rate=fee_rate
        )
        self.window = window
        self.zscore_threshold = zscore_threshold
        self.mean_reversion_threshold = mean_reversion_threshold
        self.data_fetcher = data_fetcher

    def calculate_metrics(
        self,
        prices: pd.Series
    ) -> Dict[str, Any]:
        """Calculate statistical metrics for price series."""
        try:
            # Calculate rolling statistics
            mean = prices.rolling(window=self.window).mean()
            std = prices.rolling(window=self.window).std()
            zscore = (prices - mean) / std

            # Calculate mean reversion strength
            price_changes = prices.diff()
            mean_dist = prices - mean
            mean_reversion = -np.corrcoef(
                mean_dist[:-1],
                price_changes[1:]
            )[0, 1]

            # Calculate half-life of mean reversion
            if mean_reversion > 0:
                half_life = np.log(2) / mean_reversion
            else:
                half_life = np.inf

            return {
                "mean": mean.iloc[-1],
                "std": std.iloc[-1],
                "zscore": zscore.iloc[-1],
                "mean_reversion": mean_reversion,
                "half_life": half_life
            }

        except Exception as e:
            logger.error(f"Metric calculation failed: {str(e)}")
            return {}

    def find_opportunities(
        self,
        symbol: str,
        price: float,
        metrics: Dict[str, Any]
    ) -> Optional[ArbitrageOpportunity]:
        """Find statistical arbitrage opportunities."""
        try:
            if not metrics:
                return None

            zscore = metrics["zscore"]
            mean_reversion = metrics["mean_reversion"]
            mean = metrics["mean"]

            # Check if conditions are met
            if (abs(zscore) > self.zscore_threshold and
                mean_reversion > self.mean_reversion_threshold):

                # Determine trade direction
                if zscore > 0:  # Overvalued
                    return ArbitrageOpportunity(
                        symbol=symbol,
                        buy_venue="mean",
                        sell_venue="market",
                        buy_price=mean,
                        sell_price=price,
                        timestamp=datetime.now().timestamp()
                    )
                else:  # Undervalued
                    return ArbitrageOpportunity(
                        symbol=symbol,
                        buy_venue="market",
                        sell_venue="mean",
                        buy_price=price,
                        sell_price=mean,
                        timestamp=datetime.now().timestamp()
                    )

            return None

        except Exception as e:
            logger.error(f"Opportunity finding failed: {str(e)}")
            return None

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data for statistical arbitrage opportunities."""
        try:
            if "symbol" not in data or "price" not in data:
                raise ValueError("Missing required data fields")

            symbol = data["symbol"]
            current_price = data["price"]

            if not self.data_fetcher:
                raise ValueError("No data fetcher configured")

            # Get historical data
            historical_data = self.data_fetcher(symbol)
            if historical_data is None or len(historical_data) < self.window:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "error": "Insufficient historical data"
                    }
                }

            # Calculate metrics
            metrics = self.calculate_metrics(historical_data)
            if not metrics:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "error": "Failed to calculate metrics"
                    }
                }

            # Find opportunities
            opportunity = self.find_opportunities(
                symbol,
                current_price,
                metrics
            )

            if opportunity and self.validate_opportunity(opportunity):
                self.opportunities = [opportunity]

                # Calculate confidence based on z-score and mean reversion
                zscore_conf = min(
                    abs(metrics["zscore"]) / self.zscore_threshold,
                    1.0
                )
                mr_conf = min(
                    metrics["mean_reversion"] / self.mean_reversion_threshold,
                    1.0
                )
                confidence = (zscore_conf + mr_conf) / 2

                return {
                    "action": "execute",
                    "confidence": confidence,
                    "metadata": {
                        "metrics": metrics,
                        "opportunity": opportunity.to_dict()
                    }
                }

            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {
                    "metrics": metrics,
                    "opportunities": []
                }
            }

        except Exception as e:
            logger.error(f"Statistical arbitrage processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
