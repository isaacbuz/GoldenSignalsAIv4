"""
Candlestick pattern recognition agent.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import talib
import logging
from ...base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class CandlestickPatternAgent(BaseAgent):
    """Agent that detects candlestick patterns for trading signals."""
    
    # Define pattern functions and their interpretations
    PATTERNS = {
        "CDL2CROWS": (talib.CDL2CROWS, "bearish"),
        "CDL3BLACKCROWS": (talib.CDL3BLACKCROWS, "bearish"),
        "CDL3INSIDE": (talib.CDL3INSIDE, "continuation"),
        "CDLENGULFING": (talib.CDLENGULFING, "reversal"),
        "CDLHARAMI": (talib.CDLHARAMI, "reversal"),
        "CDLMORNINGSTAR": (talib.CDLMORNINGSTAR, "bullish"),
        "CDLEVENINGSTAR": (talib.CDLEVENINGSTAR, "bearish"),
        "CDLHAMMER": (talib.CDLHAMMER, "bullish"),
        "CDLSHOOTINGSTAR": (talib.CDLSHOOTINGSTAR, "bearish")
    }
    
    def __init__(
        self,
        name: str = "CandlestickPattern",
        min_confidence: float = 0.6,
        pattern_weights: Dict[str, float] = None
    ):
        """
        Initialize candlestick pattern agent.
        
        Args:
            name: Agent name
            min_confidence: Minimum confidence for pattern signals
            pattern_weights: Custom weights for different patterns
        """
        super().__init__(name=name, agent_type="predictive")
        self.min_confidence = min_confidence
        self.pattern_weights = pattern_weights or {
            pattern: 1.0 for pattern in self.PATTERNS.keys()
        }
        
    def detect_patterns(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray
    ) -> Dict[str, int]:
        """Detect candlestick patterns in price data."""
        try:
            patterns = {}
            
            for pattern_name, (pattern_func, _) in self.PATTERNS.items():
                result = pattern_func(
                    open_prices,
                    high_prices,
                    low_prices,
                    close_prices
                )
                if result[-1] != 0:  # Check last value for pattern
                    patterns[pattern_name] = result[-1]
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {str(e)}")
            return {}
            
    def aggregate_signals(
        self,
        patterns: Dict[str, int]
    ) -> Tuple[str, float]:
        """Aggregate multiple pattern signals into a single signal."""
        try:
            if not patterns:
                return "hold", 0.0
                
            bullish_score = 0.0
            bearish_score = 0.0
            total_weight = 0.0
            
            for pattern, signal in patterns.items():
                weight = self.pattern_weights[pattern]
                pattern_type = self.PATTERNS[pattern][1]
                
                if pattern_type == "bullish" or (pattern_type == "reversal" and signal > 0):
                    bullish_score += abs(signal) * weight
                elif pattern_type == "bearish" or (pattern_type == "reversal" and signal < 0):
                    bearish_score += abs(signal) * weight
                    
                total_weight += weight
                
            if total_weight == 0:
                return "hold", 0.0
                
            # Normalize scores
            bullish_score /= total_weight
            bearish_score /= total_weight
            
            # Determine final signal
            if bullish_score > bearish_score and bullish_score >= self.min_confidence:
                return "buy", bullish_score
            elif bearish_score > bullish_score and bearish_score >= self.min_confidence:
                return "sell", bearish_score
            else:
                return "hold", max(bullish_score, bearish_score)
                
        except Exception as e:
            logger.error(f"Signal aggregation failed: {str(e)}")
            return "hold", 0.0
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and detect candlestick patterns."""
        try:
            required_fields = ["open_prices", "high_prices", "low_prices", "close_prices"]
            if not all(field in data for field in required_fields):
                raise ValueError("Missing required price data")
                
            # Convert price data to numpy arrays
            open_prices = np.array(data["open_prices"])
            high_prices = np.array(data["high_prices"])
            low_prices = np.array(data["low_prices"])
            close_prices = np.array(data["close_prices"])
            
            # Detect patterns
            patterns = self.detect_patterns(
                open_prices,
                high_prices,
                low_prices,
                close_prices
            )
            
            # Generate trading signal
            action, confidence = self.aggregate_signals(patterns)
            
            return {
                "action": action,
                "confidence": confidence,
                "metadata": {
                    "patterns": patterns,
                    "min_confidence": self.min_confidence,
                    "pattern_weights": self.pattern_weights
                }
            }
            
        except Exception as e:
            logger.error(f"Candlestick pattern analysis failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 