import joblib
import numpy as np
from archive.legacy_backend_agents.base import BaseSignalAgent

class AutoencoderAnomalyAgent(BaseSignalAgent):
    """
    ML agent that detects anomalies in price/indicator data using a trained autoencoder.
    Useful for flagging unusual market behavior or outlier events.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.model = None

    def run(self, market_data: dict) -> dict:
        try:
            self.model = joblib.load("ml_models/autoencoder_model.pkl")
            X = np.array([[
                market_data.get('open', 0),
                market_data.get('high', 0),
                market_data.get('low', 0),
                market_data.get('close', 0),
                market_data.get('volume', 0)
            ]])
            reconstructed = self.model.predict(X)
            mse = np.mean((X - reconstructed) ** 2)
            threshold = 0.01  # Ideally, this is determined from validation data
            is_anomaly = mse > threshold
            explanation = f"Anomaly score (MSE): {mse:.4f}. {'Anomaly detected!' if is_anomaly else 'Normal.'}"
            return {
                "agent": "AutoencoderAnomalyAgent",
                "anomaly_score": float(mse),
                "is_anomaly": is_anomaly,
                "explanation": explanation,
                "confidence": 95 if is_anomaly else 70
            }
        except Exception as e:
            return {
                "agent": "AutoencoderAnomalyAgent",
                "anomaly_score": None,
                "is_anomaly": False,
                "explanation": f"Could not compute anomaly score: {str(e)}",
                "confidence": 0
            }
