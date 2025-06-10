import numpy as np
from archive.legacy_backend_agents.base import BaseSignalAgent
from ml_training.feature_engineering import AdvancedFeatureEngineer

class VolatilityForecastAgent(BaseSignalAgent):
    """
    Predicts future volatility using a rolling standard deviation (can be replaced with GARCH or LSTM).
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)

    def run(self, price_history: list) -> dict:
        if len(price_history) < 10:
            return {"agent": "VolatilityForecastAgent", "volatility": None, "confidence": 0, "explanation": "Not enough data."}
        # Use AdvancedFeatureEngineer for feature extraction
        market_data = {'close': np.array(price_history)}
        features = AdvancedFeatureEngineer.extract_market_features(market_data)
        # For demonstration, use the last feature as "volatility" (or adjust as needed)
        volatility = features[-1] if len(features) > 0 else None
        explanation = f"Advanced feature-based volatility: {volatility:.4f}" if volatility is not None else "Could not compute volatility."
        return {"agent": "VolatilityForecastAgent", "volatility": float(volatility) if volatility is not None else None, "confidence": 80 if volatility is not None else 0, "explanation": explanation}
