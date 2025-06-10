import pandas as pd
from typing import Dict, Any
from ..base_agent import BaseAgent

class CandlePatternAgent(BaseAgent):
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        if not {'open', 'high', 'low', 'close'}.issubset(data.columns):
            return {
                "signal": "neutral",
                "confidence": 0,
                "explanation": "Missing OHLC data"
            }

        recent = data.iloc[-2:]
        o1, h1, l1, c1 = recent.iloc[0][['open', 'high', 'low', 'close']]
        o2, h2, l2, c2 = recent.iloc[1][['open', 'high', 'low', 'close']]

        # Bullish engulfing pattern
        if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:
            return {
                "signal": "bullish",
                "confidence": 0.8,
                "explanation": "Bullish engulfing pattern detected"
            }

        # Bearish engulfing pattern
        if c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:
            return {
                "signal": "bearish",
                "confidence": 0.8,
                "explanation": "Bearish engulfing pattern detected"
            }

        return {
            "signal": "neutral",
            "confidence": 0.4,
            "explanation": "No strong pattern detected"
        }
