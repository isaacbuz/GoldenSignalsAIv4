import numpy as np
from keras.models import load_model
from backend.agents.base import BaseSignalAgent

class PretrainedGRUStockAgent(BaseSignalAgent):
    """
    Wraps a pre-trained GRU model for stock price prediction from Stock-Prediction-Models.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.model = load_model("external/Stock-Prediction-Models/deep-learning/model/gru_model.h5")
    def run(self, price_history: list) -> dict:
        if len(price_history) < 60:
            return {"agent": "PretrainedGRUStockAgent", "error": "Not enough data (need 60+ prices)"}
        X = np.array(price_history[-60:]).reshape((1, 60, 1))
        pred = self.model.predict(X)[0][0]
        trend = "bullish" if pred > price_history[-1] else "bearish"
        return {
            "agent": "PretrainedGRUStockAgent",
            "trend": trend,
            "predicted_price": float(pred),
            "confidence": float(abs(pred - price_history[-1]) / price_history[-1]),
            "symbol": self.symbol
        }
