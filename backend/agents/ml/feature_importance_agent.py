import joblib
import numpy as np
from backend.agents.base import BaseSignalAgent

class FeatureImportanceAgent(BaseSignalAgent):
    """
    ML agent that explains which features (indicators) are driving the model's decisions using SHAP for local explanations (if available),
    or global feature importances otherwise.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.model = None
        self.feature_names = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'volatility'
        ]

    def run(self, market_data: dict) -> dict:
        try:
            self.model = joblib.load("ml_models/xgb_model.pkl")
            X = np.array([[market_data.get(n, 0) for n in self.feature_names]])
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
                abs_shap = np.abs(shap_values[0])
                top_idx = np.argsort(abs_shap)[::-1][:3]
                top_features = [(self.feature_names[i], round(abs_shap[i], 3)) for i in top_idx]
                explanation = f"SHAP local: {', '.join([f'{n} ({w})' for n, w in top_features])}"
            except ImportError:
                importances = self.model.feature_importances_
                top_idx = np.argsort(importances)[::-1][:3]
                top_features = [(self.feature_names[i], round(importances[i], 3)) for i in top_idx]
                explanation = f"Global: {', '.join([f'{n} ({w})' for n, w in top_features])}"
            return {
                "agent": "FeatureImportanceAgent",
                "top_features": top_features,
                "explanation": explanation,
                "confidence": 80
            }
        except Exception as e:
            return {
                "agent": "FeatureImportanceAgent",
                "top_features": [],
                "explanation": f"Could not compute feature importance: {str(e)}",
                "confidence": 0
            }
