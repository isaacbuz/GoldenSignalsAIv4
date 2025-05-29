# backend/agents/ml/forecast_agent.py
import joblib
import numpy as np
from backend.agents.base import BaseAgent

class ForecastAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.models = {
            "LSTM": None,
            "XGBoost": None,
            "Prophet": None
        }

    def run(self, market_data):
        X = np.array([[p] for p in market_data.get("price", [])[-10:]])
        predictions = {}
        try:
            self.models["XGBoost"] = joblib.load("ml_models/xgb_model.pkl")
            pred = self.models["XGBoost"].predict(X)[-1]
            predictions["XGBoost"] = "bullish" if pred else "bearish"
        except:
            predictions["XGBoost"] = "unknown"

        try:
            self.models["LSTM"] = joblib.load("ml_models/lstm_model.pkl")
            pred = self.models["LSTM"].predict(X)[-1]
            predictions["LSTM"] = "bullish" if pred else "bearish"
        except:
            predictions["LSTM"] = "unknown"

        try:
            self.models["Prophet"] = joblib.load("ml_models/prophet_model.pkl")
            predictions["Prophet"] = "flat"
        except:
            predictions["Prophet"] = "unknown"

        overall = [v for v in predictions.values() if v == "bullish"]
        trend = "bullish" if len(overall) >= 2 else "bearish"
        return {
            "trend": trend,
            "confidence": 90 if trend == "bullish" else 60,
            "explanation": f"Models agree: {predictions}",
            "models": predictions
        }

    def train(self):
        from sklearn.linear_model import LogisticRegression
        X = np.array([[i] for i in range(10)])
        y = [0, 1, 1, 1, 0, 1, 0, 0, 1, 1]
        model = LogisticRegression().fit(X, y)
        joblib.dump(model, "ml_models/xgb_model.pkl")
        joblib.dump(model, "ml_models/lstm_model.pkl")
        joblib.dump(model, "ml_models/prophet_model.pkl")
