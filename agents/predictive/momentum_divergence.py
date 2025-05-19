"""
momentum_divergence.py
Purpose: Implements a MomentumDivergenceAgent that detects bullish and bearish momentum divergence using a configurable momentum indicator (RSI, MOM, or MACD). Integrates with the GoldenSignalsAI agent framework.
"""

import pandas as pd
import talib
from typing import Dict, Any, Literal
import logging

from ..base_agent import BaseAgent

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

class MomentumDivergenceAgent(BaseAgent):
    """Agent that detects momentum divergence for trading signals."""

    def __init__(self, lookback: int = 20, indicator: Literal['RSI', 'MOM', 'MACD'] = 'RSI'):
        """
        Args:
            lookback (int): Number of bars to look back for divergence detection.
            indicator (str): Momentum indicator to use ('RSI', 'MOM', 'MACD').
        """
        self.lookback = lookback
        self.indicator = indicator.upper()
        logger.info({"message": f"MomentumDivergenceAgent initialized with lookback={lookback}, indicator={self.indicator}"})

    def process(self, data: Dict) -> Dict:
        """
        Process market data to detect momentum divergence signals.
        Args:
            data (Dict): Market observation with 'Close' prices (and optionally 'Volume').
        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        """
        df = pd.DataFrame(data)
        if 'Close' not in df.columns or len(df) < self.lookback + 2:
            logger.warning({"message": "Insufficient data for momentum divergence detection"})
            return {"action": "hold", "confidence": 0.0, "metadata": {}}

        divergence = self.find_momentum_divergence(df, self.lookback, self.indicator)
        signal = {"action": "hold", "confidence": 0.0, "metadata": {"indicator": self.indicator, **divergence}}
        if divergence['bullish']:
            signal.update({"action": "buy", "confidence": 1.0})
        elif divergence['bearish']:
            signal.update({"action": "sell", "confidence": 1.0})
        logger.info({"message": f"MomentumDivergenceAgent signal: {signal}"})
        return signal

    @staticmethod
    def find_momentum_divergence(df: pd.DataFrame, lookback: int, indicator: str) -> Dict[str, bool]:
        close = df['Close']
        if indicator == 'RSI':
            mom = talib.RSI(close, timeperiod=14)
        elif indicator == 'MOM':
            mom = talib.MOM(close, timeperiod=10)
        elif indicator == 'MACD':
            macd, macdsignal, macdhist = talib.MACD(close)
            mom = macd
        else:
            raise ValueError("Unsupported indicator")

        # Find recent swing lows/highs
        price_lows = close[(close.shift(1) > close) & (close.shift(-1) > close)]
        price_highs = close[(close.shift(1) < close) & (close.shift(-1) < close)]
        mom_lows = mom[(mom.shift(1) > mom) & (mom.shift(-1) > mom)]
        mom_highs = mom[(mom.shift(1) < mom) & (mom.shift(-1) < mom)]

        recent_price_lows = price_lows.tail(lookback).dropna()
        recent_mom_lows = mom_lows.tail(lookback).dropna()
        recent_price_highs = price_highs.tail(lookback).dropna()
        recent_mom_highs = mom_highs.tail(lookback).dropna()

        bullish = False
        bearish = False

        if len(recent_price_lows) >= 2 and len(recent_mom_lows) >= 2:
            # Bullish divergence: price lower low, momentum higher low
            if recent_price_lows.iloc[-1] < recent_price_lows.iloc[-2] and recent_mom_lows.iloc[-1] > recent_mom_lows.iloc[-2]:
                bullish = True

        if len(recent_price_highs) >= 2 and len(recent_mom_highs) >= 2:
            # Bearish divergence: price higher high, momentum lower high
            if recent_price_highs.iloc[-1] > recent_price_highs.iloc[-2] and recent_mom_highs.iloc[-1] < recent_mom_highs.iloc[-2]:
                bearish = True

        return {"bullish": bullish, "bearish": bearish}
