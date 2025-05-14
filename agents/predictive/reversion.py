# agents/predictive/reversion.py
# Purpose: Implements a ReversionAgent that identifies mean-reversion opportunities,
# suitable for options trading strategies like straddles in mean-reverting markets.

import logging

import pandas as pd
from typing import Dict, Any

from ..base_agent import BaseAgent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class ReversionAgent(BaseAgent):
    """Agent that identifies mean-reversion opportunities."""

    def __init__(self, mean_reversion_window: int = 20, z_score_threshold: float = 2.0):
        """Initialize the ReversionAgent.

        Args:
            mean_reversion_window (int): Lookback period for mean-reversion calculation.
            z_score_threshold (float): Threshold for z-score calculation.
        """
        self.mean_reversion_window = mean_reversion_window
        self.z_score_threshold = z_score_threshold
        logger.info(
            {
                "message": f"ReversionAgent initialized with mean_reversion_window={mean_reversion_window} and z_score_threshold={z_score_threshold}"
            }
        )

    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and potentially modify a trading signal.
        
        Args:
            signal (Dict[str, Any]): Trading signal to process.
        
        Returns:
            Dict[str, Any]: Processed trading signal with potential modifications.
        """
        # Default implementation: return signal as-is
        logger.info({"message": f"Processing signal: {signal}"})
        return signal

    def process(self, data: Dict) -> Dict:
        """Process market data to identify mean-reversion opportunities.

        Args:
            data (Dict): Market observation with 'stock_data'.

        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        """
        logger.info({"message": "Processing data for ReversionAgent"})
        try:
            stock_data = pd.DataFrame(data["stock_data"])
            if stock_data.empty:
                logger.warning({"message": "No stock data available"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            prices = stock_data["Close"]
            if len(prices) < self.mean_reversion_window:
                logger.warning(
                    {
                        "message": f"Insufficient data: {len(prices)} < {self.mean_reversion_window}"
                    }
                )
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            mean_price = prices[-self.mean_reversion_window :].mean()
            current_price = prices.iloc[-1]
            deviation = (current_price - mean_price) / mean_price

            # Detect mean-reversion opportunity
            if deviation > 0.05:
                action = "sell"  # Overbought
                confidence = deviation
            elif deviation < -0.05:
                action = "buy"  # Oversold
                confidence = abs(deviation)
            else:
                action = "hold"
                confidence = 0.0

            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {"deviation": deviation, "mean_price": mean_price},
            }
            logger.info({"message": f"ReversionAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"ReversionAgent processing failed: {str(e)}"})
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        """Adapt the agent to new market data (placeholder for learning).

        Args:
            new_data (pd.DataFrame): New market data.
        """
        logger.info({"message": "ReversionAgent adapting to new data"})
        try:
            # Placeholder: Adjust window based on volatility
            pass
        except Exception as e:
            logger.error({"message": f"ReversionAgent adaptation failed: {str(e)}"})
