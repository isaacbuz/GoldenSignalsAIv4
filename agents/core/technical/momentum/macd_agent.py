"""
MACD (Moving Average Convergence Divergence) technical analysis agent.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from ....base import BaseAgent

logger = logging.getLogger(__name__)

class MACDAgent(BaseAgent):
    """Agent that generates trading signals based on MACD crossovers and divergences."""
    
    def __init__(
        self,
        name: str = "MACD",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        divergence_threshold: float = 0.2
    ):
        """
        Initialize MACD agent.
        
        Args:
            name: Agent name
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            divergence_threshold: Minimum divergence for signal generation
        """
        super().__init__(name=name, agent_type="technical")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.divergence_threshold = divergence_threshold
        
    def calculate_macd(self, prices: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate MACD line, signal line, and histogram."""
        try:
            if len(prices) < self.slow_period + self.signal_period:
                return None, None, None
                
            # Calculate EMAs
            ema_fast = prices.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=self.slow_period, adjust=False).mean()
            
            # Calculate MACD line and signal
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return (
                float(macd_line.iloc[-1]),
                float(signal_line.iloc[-1]),
                float(histogram.iloc[-1])
            )
            
        except Exception as e:
            logger.error(f"MACD calculation failed: {str(e)}")
            return None, None, None
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate MACD signals with confidence levels."""
        try:
            if "close_prices" not in data:
                raise ValueError("Close prices not found in market data")
                
            prices = pd.Series(data["close_prices"])
            macd, signal, histogram = self.calculate_macd(prices)
            
            if macd is None:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "error": "Insufficient data for MACD calculation"
                    }
                }
            
            # Calculate signal based on MACD crossovers and histogram
            if histogram > 0 and histogram > self.divergence_threshold:
                action = "buy"
                # Confidence increases with histogram size and positive momentum
                confidence = min(histogram / self.divergence_threshold, 1.0)
            elif histogram < 0 and abs(histogram) > self.divergence_threshold:
                action = "sell"
                # Confidence increases with histogram size and negative momentum
                confidence = min(abs(histogram) / self.divergence_threshold, 1.0)
            else:
                action = "hold"
                confidence = 0.0
                
            # Adjust confidence based on trend consistency
            if len(prices) >= self.slow_period:
                trend = prices.pct_change(self.slow_period).iloc[-1]
                if (action == "buy" and trend < 0) or (action == "sell" and trend > 0):
                    confidence *= 0.8  # Reduce confidence when against trend
                
            return {
                "action": action,
                "confidence": confidence,
                "metadata": {
                    "macd": macd,
                    "signal": signal,
                    "histogram": histogram,
                    "fast_period": self.fast_period,
                    "slow_period": self.slow_period,
                    "signal_period": self.signal_period,
                    "divergence_threshold": self.divergence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"MACD signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 