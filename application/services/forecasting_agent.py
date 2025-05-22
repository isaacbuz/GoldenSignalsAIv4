import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from typing import Dict, Any

class ForecastingAgent:
    def __init__(self):
        self.model = XGBRegressor()
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict(X)
