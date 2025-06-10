import pandas as pd
from typing import Dict, Any
from ..base_agent import BaseAgent

class OBVAgent(BaseAgent):
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        if not {'close', 'volume'}.issubset(data.columns):
            return {
                "signal": "neutral",
                "confidence": 0.0,
                "explanation": "Missing close or volume data"
            }

        obv = [0]
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i - 1]:
                obv.append(obv[-1] + data['volume'].iloc[i])
            elif data['close'].iloc[i] < data['close'].iloc[i - 1]:
                obv.append(obv[-1] - data['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        obv_series = pd.Series(obv)
        recent = obv_series.iloc[-5:]
        slope = recent.diff().mean()

        if slope > 0:
            return {
                "signal": "bullish",
                "confidence": min(slope / recent.abs().mean(), 1.0),
                "explanation": f"OBV rising, buyers accumulating (slope = {slope:.2f})"
            }
        elif slope < 0:
            return {
                "signal": "bearish",
                "confidence": min(-slope / recent.abs().mean(), 1.0),
                "explanation": f"OBV falling, selling pressure detected (slope = {slope:.2f})"
            }

        return {
            "signal": "neutral",
            "confidence": 0.3,
            "explanation": "OBV flat, no directional signal"
        }
