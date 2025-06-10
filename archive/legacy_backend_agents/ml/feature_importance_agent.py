import joblib
import numpy as np
from archive.legacy_backend_agents.base import BaseSignalAgent
from ml_training.feature_engineering import AdvancedFeatureEngineer

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
            # Use AdvancedFeatureEngineer to extract features from market_data
            features = AdvancedFeatureEngineer.extract_market_features(market_data)
            X = features.reshape(1, -1)
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
                abs_shap = np.abs(shap_values[0])
                # Use generic feature names for now
                feature_names = [f"f{i}" for i in range(X.shape[1])]
                top_idx = np.argsort(abs_shap)[::-1][:3]
                top_features = [(feature_names[i], round(abs_shap[i], 3)) for i in top_idx]
                explanation = f"SHAP local: {', '.join([f'{n} ({w})' for n, w in top_features])}"
            except ImportError:
                importances = self.model.feature_importances_
                feature_names = [f"f{i}" for i in range(X.shape[1])]
                top_idx = np.argsort(importances)[::-1][:3]
                top_features = [(feature_names[i], round(importances[i], 3)) for i in top_idx]
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
