import numpy as np
from keras.models import load_model
from backend.agents.base import BaseSignalAgent

class LSTMPriceForecastAgent(BaseSignalAgent):
    """
    Uses a trained LSTM neural network to forecast future prices or trends.
    Expects a Keras model saved as 'ml_models/lstm_price_model.h5'.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.model = load_model("ml_models/lstm_price_model.h5")

    def run(self, price_history: list) -> dict:
        if len(price_history) < 20:
            return {"agent": "LSTMPriceForecastAgent", "trend": None, "confidence": 0, "explanation": "Not enough data."}
        X = np.array(price_history[-20:]).reshape((1, 20, 1))
        pred = self.model.predict(X)[0][0]
        trend = "bullish" if pred > price_history[-1] else "bearish"
        explanation = f"LSTM predicts next price: {pred:.2f} (current: {price_history[-1]:.2f})"
        return {"agent": "LSTMPriceForecastAgent", "trend": trend, "predicted_price": float(pred), "confidence": 85, "explanation": explanation}
