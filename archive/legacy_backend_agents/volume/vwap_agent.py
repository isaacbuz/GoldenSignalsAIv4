import pandas as pd
from typing import Dict, Any
from ..base_agent import BaseAgent

class VWAPAgent(BaseAgent):
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        if not {'close', 'high', 'low', 'volume'}.issubset(data.columns):
            return {
                "signal": "neutral",
                "confidence": 0.0,
                "explanation": "Missing required price/volume fields"
            }

        df = data.copy()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tpv'] = df['typical_price'] * df['volume']
        cumulative_tpv = df['tpv'].cumsum()
        cumulative_volume = df['volume'].cumsum()
        df['vwap'] = cumulative_tpv / cumulative_volume

        latest_price = df['close'].iloc[-1]
        latest_vwap = df['vwap'].iloc[-1]
        distance = latest_price - latest_vwap

        if distance > 0:
            return {
                "signal": "bullish",
                "confidence": min(distance / latest_vwap, 1.0),
                "explanation": f"Price above VWAP by {distance:.2f}"
            }
        elif distance < 0:
            return {
                "signal": "bearish",
                "confidence": min(-distance / latest_vwap, 1.0),
                "explanation": f"Price below VWAP by {abs(distance):.2f}"
            }

        return {
            "signal": "neutral",
            "confidence": 0.3,
            "explanation": "Price equal to VWAP"
        }
