import logging
from typing import Any, Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

class MLClassifierAgent:
    """Agent for ML-based classification with RandomForest/XGBoost and robust error handling."""
    def __init__(self, model_type: str = 'random_forest', model_path: Optional[str] = None):
        if model_type == 'xgboost':
            self.model = XGBClassifier()
        else:
            self.model = RandomForestClassifier(n_estimators=100)
        self.model_path = model_path
        self._trained = False
        try:
            if model_path:
                self.load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.model = None

    def load_model(self, path: str):
        self.model = joblib.load(path)
        self._trained = True

    def train(self, features: pd.DataFrame, labels: pd.Series):
        self.model.fit(features, labels)
        self._trained = True

    def predict_signal(self, features: pd.DataFrame) -> Any:
        """Predict trading signal from features. Returns 'buy', 'sell', 'hold', or error dict."""
        if not hasattr(self.model, 'predict_proba') or not self._trained:
            logger.error("ML model not trained or loaded.")
            return {"error": "ML model not trained or loaded."}
        if not isinstance(features, pd.DataFrame) or features.empty:
            logger.warning("Invalid or empty features DataFrame.")
            return {"error": "Invalid or empty features DataFrame."}
        try:
            prob = self.model.predict_proba(features)[-1]
            if prob[1] > 0.6:
                return "buy"
            elif prob[1] < 0.4:
                return "sell"
            return "hold"
        except Exception as e:
            logger.error(f"MLClassifier prediction failed: {e}")
            return {"error": str(e)}
