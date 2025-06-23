"""
Options chain analysis agent.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
from ...base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class OptionsChainAgent(BaseAgent):
    """Agent that analyzes options chain data for trading signals."""
    
    def __init__(
        self,
        name: str = "OptionsChain",
        min_volume: int = 100,
        min_open_interest: int = 500,
        volatility_window: int = 20,
        skew_threshold: float = 0.15
    ):
        """
        Initialize options chain agent.
        
        Args:
            name: Agent name
            min_volume: Minimum option volume for analysis
            min_open_interest: Minimum open interest for analysis
            volatility_window: Window for historical volatility
            skew_threshold: Threshold for skew significance
        """
        super().__init__(name=name, agent_type="predictive")
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest
        self.volatility_window = volatility_window
        self.skew_threshold = skew_threshold
        
    def calculate_implied_volatility_skew(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate implied volatility skew metrics."""
        try:
            # Filter by volume and open interest
            calls = calls[
                (calls["volume"] >= self.min_volume) &
                (calls["open_interest"] >= self.min_open_interest)
            ]
            puts = puts[
                (puts["volume"] >= self.min_volume) &
                (puts["open_interest"] >= self.min_open_interest)
            ]
            
            if calls.empty or puts.empty:
                return {"skew": 0.0, "call_put_ratio": 1.0}
                
            # Calculate IV skew
            atm_call_iv = calls[calls["moneyness"].abs().idxmin()]["implied_volatility"]
            atm_put_iv = puts[puts["moneyness"].abs().idxmin()]["implied_volatility"]
            
            skew = (atm_put_iv - atm_call_iv) / ((atm_put_iv + atm_call_iv) / 2)
            
            # Calculate volume-weighted call/put ratio
            call_volume = (calls["volume"] * calls["open_interest"]).sum()
            put_volume = (puts["volume"] * puts["open_interest"]).sum()
            
            call_put_ratio = call_volume / put_volume if put_volume > 0 else 1.0
            
            return {
                "skew": float(skew),
                "call_put_ratio": float(call_put_ratio)
            }
            
        except Exception as e:
            logger.error(f"IV skew calculation failed: {str(e)}")
            return {"skew": 0.0, "call_put_ratio": 1.0}
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process options chain data and generate trading signals."""
        try:
            if "calls" not in data or "puts" not in data:
                raise ValueError("Options chain data not found")
                
            calls = pd.DataFrame(data["calls"])
            puts = pd.DataFrame(data["puts"])
            
            # Calculate IV skew metrics
            metrics = self.calculate_implied_volatility_skew(calls, puts)
            
            # Generate signal based on skew and volume
            if abs(metrics["skew"]) < self.skew_threshold:
                action = "hold"
                confidence = 0.0
            elif metrics["skew"] > self.skew_threshold:
                # Positive skew (puts more expensive) - bearish
                action = "sell"
                confidence = min(abs(metrics["skew"]) / self.skew_threshold, 1.0)
            else:
                # Negative skew (calls more expensive) - bullish
                action = "buy"
                confidence = min(abs(metrics["skew"]) / self.skew_threshold, 1.0)
                
            # Adjust confidence based on call/put ratio
            if metrics["call_put_ratio"] > 1.5:
                confidence *= 1.2  # Boost confidence for heavy call activity
            elif metrics["call_put_ratio"] < 0.5:
                confidence *= 1.2  # Boost confidence for heavy put activity
                
            return {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {
                    "iv_skew": metrics["skew"],
                    "call_put_ratio": metrics["call_put_ratio"],
                    "min_volume": self.min_volume,
                    "min_open_interest": self.min_open_interest,
                    "skew_threshold": self.skew_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Options chain analysis failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 