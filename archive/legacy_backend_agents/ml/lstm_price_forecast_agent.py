import numpy as np
from keras.models import load_model
from archive.legacy_backend_agents.base import BaseSignalAgent
from ml_training.feature_engineering import AdvancedFeatureEngineer

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
        # Use AdvancedFeatureEngineer for feature extraction
        market_data = {'close': np.array(price_history)}
        features = AdvancedFeatureEngineer.extract_market_features(market_data)
        X = features.reshape((1, -1, 1))
        pred = self.model.predict(X)[0][0]
        trend = "bullish" if pred > price_history[-1] else "bearish"
        explanation = f"LSTM predicts next price: {pred:.2f} (current: {price_history[-1]:.2f})"
        return {"agent": "LSTMPriceForecastAgent", "trend": trend, "predicted_price": float(pred), "confidence": 85, "explanation": explanation}
