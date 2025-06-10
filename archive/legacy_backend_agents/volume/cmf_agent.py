import pandas as pd
from typing import Dict, Any
from ..base_agent import BaseAgent

class CMFAgent(BaseAgent):
    def __init__(self, period: int = 20):
        self.period = period

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        if not {'high', 'low', 'close', 'volume'}.issubset(data.columns):
            return {
                "signal": "neutral",
                "confidence": 0.0,
                "explanation": "Missing required OHLCV data"
            }

        df = data.copy()
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 0.01)
        mf_volume = clv * df['volume']
        cmf = mf_volume.rolling(self.period).sum() / df['volume'].rolling(self.period).sum()
        latest_cmf = cmf.iloc[-1]

        if latest_cmf > 0.1:
            return {
                "signal": "bullish",
                "confidence": min(latest_cmf, 1.0),
                "explanation": f"CMF = {latest_cmf:.2f}  strong buying pressure"
            }
        elif latest_cmf < -0.1:
            return {
                "signal": "bearish",
                "confidence": min(-latest_cmf, 1.0),
                "explanation": f"CMF = {latest_cmf:.2f}  strong selling pressure"
            }

        return {
            "signal": "neutral",
            "confidence": 0.4,
            "explanation": f"CMF = {latest_cmf:.2f}  no dominant pressure"
        }
