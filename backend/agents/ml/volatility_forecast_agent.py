import numpy as np
from backend.agents.base import BaseSignalAgent

class VolatilityForecastAgent(BaseSignalAgent):
    """
    Predicts future volatility using a rolling standard deviation (can be replaced with GARCH or LSTM).
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)

    def run(self, price_history: list) -> dict:
        if len(price_history) < 10:
            return {"agent": "VolatilityForecastAgent", "volatility": None, "confidence": 0, "explanation": "Not enough data."}
        returns = np.diff(np.log(price_history))
        vol = np.std(returns[-10:]) * np.sqrt(252)  # annualized
        explanation = f"10-day rolling volatility: {vol:.4f}"
        return {"agent": "VolatilityForecastAgent", "volatility": float(vol), "confidence": 80, "explanation": explanation}
