import numpy as np
import pandas as pd

class VolatilityAgent:
    def __init__(self, window=14):
        self.window = window

    def analyze(self, market_data: pd.DataFrame) -> dict:
        high = market_data['High']
        low = market_data['Low']
        close = market_data['Close']
        tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
        atr = tr.rolling(window=self.window).mean().iloc[-1]
        confidence = min(1.0, atr / (close.iloc[-1] * 0.05))
        return {
            "label": "Volatility",
            "value": f"ATR({self.window})={atr:.2f}",
            "confidence": round(confidence * 100, 2),
            "rationale": f"ATR over last {self.window} periods is {atr:.2f}."
        } 