"""
XGBoost-based forecasting agent for market prediction.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from typing import Dict, Any

class ForecastingAgent:
    """
    A forecasting agent that uses XGBoost for market prediction.
    """
    def __init__(self):
        self.model = XGBRegressor()
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the forecasting model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
        """
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Feature matrix for prediction
            
        Returns:
            np.ndarray: Predicted values
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict(X) 