# domain/trading/indicators.py
# Purpose: Computes technical indicators (e.g., RSI, MACD) and adjusts signals based on
# market regime detection (trending, mean-reverting, volatile). Enhanced for options trading
# to support regime-adjusted strategies.

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class Indicators:
    """Computes technical indicators and detects market regimes."""

    def __init__(
        self,
        stock_data: pd.DataFrame,
        intraday_data: pd.DataFrame = None,
        weekly_data: pd.DataFrame = None,
    ):
        """Initialize with multi-timeframe data.

        Args:
            stock_data (pd.DataFrame): Daily stock data.
            intraday_data (pd.DataFrame, optional): Intraday data.
            weekly_data (pd.DataFrame, optional): Weekly data.
        """
        self.stock_data = stock_data
        self.intraday_data = intraday_data
        self.weekly_data = weekly_data
        logger.info({"message": "Indicators initialized"})

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index (RSI).

        Args:
            prices (pd.Series): Price series.
            period (int): Lookback period (default: 14).

        Returns:
            pd.Series: RSI values.
        """
        logger.debug({"message": f"Computing RSI with period={period}"})
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error({"message": f"Failed to compute RSI: {str(e)}"})
            return pd.Series(index=prices.index)

    def compute_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.Series:
        """Compute Moving Average Convergence Divergence (MACD).

        Args:
            prices (pd.Series): Price series.
            fast (int): Fast EMA period (default: 12).
            slow (int): Slow EMA period (default: 26).
            signal (int): Signal line period (default: 9).

        Returns:
            pd.Series: MACD line values.
        """
        logger.debug(
            {
                "message": f"Computing MACD with fast={fast}, slow={slow}, signal={signal}"
            }
        )
        try:
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            return macd
        except Exception as e:
            logger.error({"message": f"Failed to compute MACD: {str(e)}"})
            return pd.Series(index=prices.index)

    def detect_market_regime(self) -> str:
        """Detect the current market regime based on multi-timeframe data.

        Returns:
            str: Market regime ('trending', 'mean_reverting', 'volatile').
        """
        logger.info({"message": "Detecting market regime"})
        try:
            # Simplified regime detection based on volatility and trend
            prices = self.stock_data["close"]
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            trend = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20]

            if volatility > 0.3:
                regime = "volatile"
            elif abs(trend) > 0.05:
                regime = "trending"
            else:
                regime = "mean_reverting"

            logger.info({"message": f"Detected market regime: {regime}"})
            return regime
        except Exception as e:
            logger.error({"message": f"Failed to detect market regime: {str(e)}"})
            return "mean_reverting"

    def compute_regime_adjusted_signal(self, indicators: List[str]) -> Dict[str, float]:
        """Compute regime-adjusted signals for selected indicators.

        Args:
            indicators (List[str]): List of indicators to compute (e.g., ['RSI', 'MACD']).

        Returns:
            Dict[str, float]: Dictionary of indicator signals adjusted for regime.
        """
        logger.info({"message": f"Computing regime-adjusted signals for {indicators}"})
        try:
            regime = self.detect_market_regime()
            signals = {}
            prices = self.stock_data["close"]

            for indicator in indicators:
                if indicator == "RSI":
                    rsi = self.compute_rsi(prices)
                    signal = -1 if rsi.iloc[-1] > 70 else 1 if rsi.iloc[-1] < 30 else 0
                elif indicator == "MACD":
                    macd = self.compute_macd(prices)
                    signal = 1 if macd.iloc[-1] > 0 else -1 if macd.iloc[-1] < 0 else 0
                else:
                    signal = 0

                # Adjust signal based on regime for options trading
                if regime == "trending":
                    signals[indicator] = (
                        signal * 1.2
                    )  # Amplify in trending markets (directional options)
                elif regime == "mean_reverting":
                    signals[indicator] = (
                        signal * 1.5
                    )  # Amplify in mean-reverting markets (straddles)
                else:  # volatile
                    signals[indicator] = (
                        signal * 0.8
                    )  # Reduce in volatile markets (hedging)

            logger.info({"message": f"Computed signals: {signals}"})
            return signals
        except Exception as e:
            logger.error(
                {"message": f"Failed to compute regime-adjusted signals: {str(e)}"}
            )
            return {}
