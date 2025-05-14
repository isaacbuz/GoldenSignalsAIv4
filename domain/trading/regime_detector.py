# domain/trading/regime_detector.py
# Purpose: Detects market regimes (trending, mean-reverting, volatile) based on price
# and volatility patterns. Used by agents to adjust trading strategies, particularly
# for options trading where regime impacts strategy selection.

import logging

import numpy as np
import pandas as pd

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects market regimes based on price and volatility data."""

    def __init__(self, window: int = 20):
        """Initialize with a lookback window.

        Args:
            window (int): Lookback period for regime detection (default: 20).
        """
        self.window = window
        logger.info({"message": f"RegimeDetector initialized with window={window}"})

    def detect(self, prices: pd.Series) -> str:
        """Detect the market regime based on price data.

        Args:
            prices (pd.Series): Price series.

        Returns:
            str: Market regime ('trending', 'mean_reverting', 'volatile').
        """
        logger.info(
            {"message": f"Detecting regime for price series with length={len(prices)}"}
        )
        try:
            if len(prices) < self.window:
                logger.warning(
                    {
                        "message": f"Insufficient data for regime detection: {len(prices)} < {self.window}"
                    }
                )
                return "mean_reverting"

            # Calculate returns and volatility
            returns = prices.pct_change().dropna()
            volatility = returns[-self.window :].std() * np.sqrt(252)
            # Calculate trend strength
            trend = (prices.iloc[-1] - prices.iloc[-self.window]) / prices.iloc[
                -self.window
            ]

            # Determine regime for options trading
            if volatility > 0.3:
                regime = "volatile"  # Suitable for volatility-based options strategies
            elif abs(trend) > 0.05:
                regime = "trending"  # Suitable for directional options plays
            else:
                regime = (
                    "mean_reverting"  # Suitable for mean-reversion options strategies
                )

            logger.info(
                {
                    "message": f"Detected regime: {regime}",
                    "volatility": volatility,
                    "trend": trend,
                }
            )
            return regime
        except Exception as e:
            logger.error({"message": f"Failed to detect regime: {str(e)}"})
            return "mean_reverting"
