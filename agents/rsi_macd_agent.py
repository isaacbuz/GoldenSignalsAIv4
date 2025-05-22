import logging
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

class RSIMACDAgent:
    """Agent for RSI and MACD-based signal generation with robust error handling."""
    def __init__(self, rsi_period: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def compute_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def compute_macd(self, prices: pd.Series) -> Any:
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        return macd_line, signal_line

    def compute_signal(self, df: pd.DataFrame) -> Any:
        """Compute trading signal based on RSI and MACD. Returns 'buy', 'sell', 'hold', or error dict."""
        if not isinstance(df, pd.DataFrame) or 'close' not in df:
            logger.warning("Input DataFrame invalid or missing 'close' column.")
            return {"error": "Input DataFrame invalid or missing 'close' column."}
        if len(df) < max(self.rsi_period, self.macd_slow):
            logger.warning("Not enough data for RSI/MACD calculation.")
            return {"error": f"Not enough data for RSI/MACD (need at least {max(self.rsi_period, self.macd_slow)} rows)."}
        try:
            close = df["close"]
            df["rsi"] = self.compute_rsi(close)
            df["macd"], df["macd_signal"] = self.compute_macd(close)
            latest = df.iloc[-1]
            if latest["rsi"] < 30 and latest["macd"] > latest["macd_signal"]:
                return "buy"
            elif latest["rsi"] > 70 and latest["macd"] < latest["macd_signal"]:
                return "sell"
            return "hold"
        except Exception as e:
            logger.error(f"RSI/MACD signal computation failed: {e}")
            return {"error": str(e)}
