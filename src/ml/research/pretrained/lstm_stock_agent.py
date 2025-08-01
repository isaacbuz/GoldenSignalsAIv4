"""
LSTM-based stock prediction agent using pretrained model.
"""
from typing import Any, Dict

import numpy as np
from agents.ml.pretrained.base_pretrained_agent import BasePretrainedAgent


class LSTMStockAgent(BasePretrainedAgent):
    """Agent using pretrained LSTM model for stock prediction."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize LSTM stock agent.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(
            name="lstm_stock",
            model_name="lstm_stock",
            model_path="external/Stock-Prediction-Models/deep-learning/model/lstm_model.h5",
            config=config,
        )

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data has required fields and length.

        Args:
            data: Input data dictionary

        Returns:
            True if valid, False otherwise
        """
        if "close" not in data:
            return False

        prices = data["close"]
        if len(prices) < 60:  # Minimum required sequence length
            return False

        return True

    def _prepare_input(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare price data for LSTM model.

        Args:
            data: Input data dictionary

        Returns:
            Numpy array of shape (1, 60, 1)
        """
        prices = np.array(data["close"])
        # Use last 60 prices, scaled to model's expected range
        sequence = prices[-60:]
        # Scale to [0, 1] range
        sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min())
        return sequence.reshape((1, 60, 1))

    def _format_output(self, prediction: np.ndarray, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format model prediction into trading signal.

        Args:
            prediction: Raw model prediction
            data: Original input data

        Returns:
            Dictionary with trading signal and metadata
        """
        current_price = data["close"][-1]
        predicted_price = float(prediction[0][0])

        # Calculate trend and confidence
        price_change = predicted_price - current_price
        trend = "bullish" if price_change > 0 else "bearish"
        confidence = float(abs(price_change) / current_price)

        return {
            "signal": trend,
            "confidence": confidence,
            "predicted_price": predicted_price,
            "current_price": current_price,
            "price_change": price_change,
            "price_change_pct": price_change / current_price * 100,
        }
