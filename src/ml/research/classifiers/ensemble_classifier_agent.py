"""
Ensemble classifier agent combining multiple ML models for robust market predictions.
"""
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from agents.common.base_agent import BaseAgent
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class EnsembleClassifierAgent(BaseAgent):
    """Agent that combines multiple ML classifiers for market prediction."""

    def __init__(
        self,
        name: str = "ensemble_classifier",
        config: Optional[Dict[str, Any]] = None,
        models_dir: str = "models",
    ):
        """
        Initialize ensemble classifier agent.

        Args:
            name: Agent identifier
            config: Configuration dictionary
            models_dir: Directory containing saved models
        """
        super().__init__(name, config)
        self.models_dir = models_dir
        self.models = {}
        self.scaler = StandardScaler()
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize ML models with default parameters."""
        self.models["rf"] = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.models["xgb"] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.models["lgb"] = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )

    def _prepare_features(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Prepare feature matrix from input data.

        Args:
            data: Input market data

        Returns:
            DataFrame of features or None if invalid input
        """
        try:
            df = pd.DataFrame(data)
            if "close" not in df.columns:
                self.logger.error("Missing required price data")
                return None

            # Calculate technical features
            df["returns"] = df["close"].pct_change()
            df["volatility"] = df["returns"].rolling(20).std()
            df["sma_20"] = df["close"].rolling(20).mean()
            df["sma_50"] = df["close"].rolling(50).mean()

            # Add volume features if available
            if "volume" in df.columns:
                df["volume_sma"] = df["volume"].rolling(20).mean()
                df["volume_ratio"] = df["volume"] / df["volume_sma"]

            # Drop missing values and select features
            feature_cols = [c for c in df.columns if c not in ["close", "returns"]]
            features = df[feature_cols].dropna()

            return features

        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return None

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictions using ensemble of models.

        Args:
            data: Market data dictionary

        Returns:
            Dictionary containing predictions and metadata
        """
        if not self.validate_input(data):
            return {"error": "Invalid input data"}

        features = self._prepare_features(data)
        if features is None:
            return {"error": "Feature preparation failed"}

        try:
            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Get predictions from each model
            predictions = {}
            probabilities = []

            for name, model in self.models.items():
                pred_proba = model.predict_proba(features_scaled)[-1]
                predictions[f"{name}_probability"] = float(pred_proba[1])
                probabilities.append(pred_proba[1])

            # Ensemble decision
            avg_probability = np.mean(probabilities)
            if avg_probability > 0.6:
                signal = "buy"
            elif avg_probability < 0.4:
                signal = "sell"
            else:
                signal = "hold"

            return {
                "signal": signal,
                "confidence": float(avg_probability),
                "model_predictions": predictions,
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data contains required fields.

        Args:
            data: Input data dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["close"]
        return all(field in data for field in required_fields)

    def save_models(self) -> None:
        """Save trained models to disk."""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            for name, model in self.models.items():
                path = os.path.join(self.models_dir, f"{self.name}_{name}_model.joblib")
                joblib.dump(model, path)
            # Save scaler
            scaler_path = os.path.join(self.models_dir, f"{self.name}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")

    def load_models(self) -> None:
        """Load trained models from disk."""
        try:
            for name in self.models.keys():
                path = os.path.join(self.models_dir, f"{self.name}_{name}_model.joblib")
                if os.path.exists(path):
                    self.models[name] = joblib.load(path)
            # Load scaler
            scaler_path = os.path.join(self.models_dir, f"{self.name}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
