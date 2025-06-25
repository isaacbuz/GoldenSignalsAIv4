"""
RSI (Relative Strength Index) technical analysis agent with advanced features.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import logging
from ....base import BaseAgent

logger = logging.getLogger(__name__)

class RSIAgent(BaseAgent):
    """Agent that generates trading signals based on RSI with trend-adjusted thresholds."""
    
    def __init__(
        self,
        name: str = "RSI",
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        trend_factor: bool = True
    ):
        """
        Initialize RSI agent.
        
        Args:
            name: Agent name
            period: RSI calculation period
            overbought: Overbought threshold
            oversold: Oversold threshold
            trend_factor: Whether to adjust thresholds based on trend
        """
        super().__init__(name=name, agent_type="technical")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.trend_factor = trend_factor
        
    def calculate_rsi(self, prices: pd.Series) -> Optional[float]:
        """Calculate RSI with trend-adjusted thresholds."""
        try:
            if len(prices) < self.period + 1:
                return None
                
            # Calculate returns and separate gains/losses
            deltas = prices.diff()
            gains = deltas.where(deltas > 0, 0.0)
            losses = -deltas.where(deltas < 0, 0.0)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=self.period).mean()
            avg_loss = losses.rolling(window=self.period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            
            # Adjust thresholds based on trend if enabled
            if self.trend_factor:
                trend = (prices.iloc[-1] / prices.iloc[-self.period] - 1) * 100
                self.overbought = min(80, 70 + abs(trend))
                self.oversold = max(20, 30 - abs(trend))
            
            return float(rsi.iloc[-1])
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {str(e)}")
            return None
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate RSI signals with confidence levels."""
        try:
            if "close_prices" not in data:
                raise ValueError("Close prices not found in market data")
                
            prices = pd.Series(data["close_prices"])
            rsi = self.calculate_rsi(prices)
            
            if rsi is None:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "rsi": None,
                        "error": "Insufficient data for RSI calculation"
                    }
                }
            
            # Calculate confidence based on RSI distance from thresholds
            if rsi > self.overbought:
                action = "sell"
                confidence = min((rsi - self.overbought) / (100 - self.overbought), 1.0)
            elif rsi < self.oversold:
                action = "buy"
                confidence = min((self.oversold - rsi) / self.oversold, 1.0)
            else:
                action = "hold"
                # Confidence decreases as RSI approaches middle range
                mid_point = (self.overbought + self.oversold) / 2
                confidence = 1.0 - abs(rsi - mid_point) / (mid_point - self.oversold)
                
            # Adjust confidence based on trend consistency
            if len(prices) >= self.period:
                trend = prices.pct_change(self.period).iloc[-1]
                if (action == "buy" and trend > 0) or (action == "sell" and trend < 0):
                    confidence *= 0.8  # Reduce confidence when against trend
                
            return {
                "action": action,
                "confidence": max(min(confidence, 1.0), 0.0),
                "metadata": {
                    "rsi": rsi,
                    "period": self.period,
                    "overbought": self.overbought,
                    "oversold": self.oversold,
                    "trend_adjusted": self.trend_factor
                }
            }
            
        except Exception as e:
            logger.error(f"RSI signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Process and potentially modify a trading signal."""
        # Default implementation: return signal as-is
        return signal 