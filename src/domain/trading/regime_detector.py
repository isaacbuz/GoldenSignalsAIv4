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


import numpy as np
import pandas as pd


class MarketRegimeDetector:
    def __init__(self, volatility_window: int = 14, threshold: float = 0.02):
        self.volatility_window = volatility_window
        self.threshold = threshold

    def detect_regime(self, prices: pd.Series) -> str:
        if len(prices) < self.volatility_window:
            return "unknown"

        returns = prices.pct_change().dropna()
        rolling_vol = returns.rolling(self.volatility_window).std().iloc[-1]

        if rolling_vol > self.threshold * 2:
            return "bear"
        elif rolling_vol < self.threshold:
            return "bull"
        else:
            return "sideways"

    def apply_to_series(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        df = df.copy()
        df["regime"] = df[price_col].rolling(self.volatility_window).apply(
            lambda x: self._map_vol_to_regime(np.std(np.diff(np.log(x)))), raw=False
        )
        return df

    def _map_vol_to_regime(self, vol: float) -> int:
        if vol > self.threshold * 2:
            return 2  # bear
        elif vol < self.threshold:
            return 0  # bull
        return 1  # sideways
