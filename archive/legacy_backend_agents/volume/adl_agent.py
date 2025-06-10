import pandas as pd
from typing import Dict, Any
from ..base_agent import BaseAgent

class ADLAgent(BaseAgent):
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        if not {'high', 'low', 'close', 'volume'}.issubset(data.columns):
            return {
                "signal": "neutral",
                "confidence": 0.0,
                "explanation": "Missing OHLCV data"
            }

        df = data.copy()
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 0.01)
        adl = (clv * df['volume']).cumsum()
        recent = adl.iloc[-5:]
        slope = recent.diff().mean()

        if slope > 0:
            return {
                "signal": "bullish",
                "confidence": min(slope / recent.abs().mean(), 1.0),
                "explanation": f"ADL rising, accumulation in progress (slope = {slope:.2f})"
            }
        elif slope < 0:
            return {
                "signal": "bearish",
                "confidence": min(-slope / recent.abs().mean(), 1.0),
                "explanation": f"ADL falling, distribution detected (slope = {slope:.2f})"
            }

        return {
            "signal": "neutral",
            "confidence": 0.3,
            "explanation": "ADL flat, no accumulation or distribution"
        }
