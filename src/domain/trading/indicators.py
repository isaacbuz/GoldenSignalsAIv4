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


class TechnicalIndicators:
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
        """Detect the current market regime based on price action and volatility."""
        try:
            # Calculate volatility
            returns = self.stock_data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate trend strength using ADX
            high = self.stock_data['high']
            low = self.stock_data['low']
            close = self.stock_data['close']
            
            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = low.diff()
            
            # Calculate TR (True Range)
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(window=14).mean()
            
            # Trend strength
            trend_strength = atr.iloc[-1] / close.iloc[-1]
            
            if volatility > 0.25:  # High volatility threshold
                return "volatile"
            elif trend_strength > 0.02:  # Strong trend threshold
                return "trending"
            else:
                return "mean_reverting"
                
        except Exception as e:
            logger.error(f"Failed to detect market regime: {str(e)}")
            return "unknown"

    def compute_regime_adjusted_signal(self, indicators: list) -> dict:
        """Compute regime-adjusted signals for selected indicators."""
        try:
            regime = self.detect_market_regime()
            signals = {}
            
            for indicator in indicators:
                if indicator == "RSI":
                    rsi = self.compute_rsi(self.stock_data['close'])
                    signal = -1 if rsi.iloc[-1] > 70 else 1 if rsi.iloc[-1] < 30 else 0
                elif indicator == "MACD":
                    macd_line = self.compute_macd(self.stock_data['close'])
                    signal_line = self.compute_macd(self.stock_data['close'], 12, 26, 9)
                    signal = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
                
                # Adjust signal based on regime
                if regime == "trending":
                    signals[indicator] = signal * 1.2  # Amplify in trending markets
                elif regime == "mean_reverting":
                    signals[indicator] = signal * 1.5  # Amplify in mean-reverting markets
                else:  # volatile
                    signals[indicator] = signal * 0.8  # Reduce in volatile markets
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to compute regime-adjusted signals: {str(e)}")
            return {}
