import pandas as pd
from typing import Dict, Any
from ..base_agent import BaseAgent

class StructureBreakAgent(BaseAgent):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        if len(data) < self.lookback + 1 or 'close' not in data.columns:
            return {
                "signal": "neutral",
                "confidence": 0,
                "explanation": "Not enough data"
            }

        highs = data['close'].rolling(self.lookback).max()
        lows = data['close'].rolling(self.lookback).min()

        recent_close = data['close'].iloc[-1]
        prior_high = highs.iloc[-2]
        prior_low = lows.iloc[-2]

        if recent_close > prior_high:
            return {
                "signal": "bullish",
                "confidence": 0.85,
                "explanation": "Structure break above prior high"
            }

        if recent_close < prior_low:
            return {
                "signal": "bearish",
                "confidence": 0.85,
                "explanation": "Structure break below prior low"
            }

        return {
            "signal": "neutral",
            "confidence": 0.3,
            "explanation": "No structure break detected"
        }
