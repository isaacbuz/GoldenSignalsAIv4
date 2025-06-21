"""
Bollinger Bands Agent
Detects volatility changes and mean reversion opportunities
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BollingerBandsAgent:
    """Bollinger Bands volatility and mean reversion agent"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.name = "bollinger_bands_agent"
        self.period = period
        self.std_dev = std_dev
        
    def calculate_bands(self, prices: pd.Series) -> tuple:
        """Calculate Bollinger Bands"""
        # Middle Band - Simple Moving Average
        middle_band = prices.rolling(window=self.period).mean()
        
        # Standard Deviation
        std = prices.rolling(window=self.period).std()
        
        # Upper and Lower Bands
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        return upper_band, middle_band, lower_band
    
    def calculate_bandwidth(self, upper: pd.Series, lower: pd.Series, middle: pd.Series) -> pd.Series:
        """Calculate Bollinger Band Width (volatility indicator)"""
        return (upper - lower) / middle
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal based on Bollinger Bands"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            
            if data.empty or len(data) < self.period + 1:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data", "agent": self.name}
                }
            
            # Calculate Bollinger Bands
            upper, middle, lower = self.calculate_bands(data['Close'])
            bandwidth = self.calculate_bandwidth(upper, lower, middle)
            
            # Current values
            current_price = data['Close'].iloc[-1]
            current_upper = upper.iloc[-1]
            current_middle = middle.iloc[-1]
            current_lower = lower.iloc[-1]
            current_bandwidth = bandwidth.iloc[-1]
            
            # Previous values for comparison
            prev_price = data['Close'].iloc[-2]
            prev_upper = upper.iloc[-2]
            prev_lower = lower.iloc[-2]
            
            # Calculate position within bands (0 = lower, 1 = upper)
            band_position = (current_price - current_lower) / (current_upper - current_lower) if current_upper != current_lower else 0.5
            
            # Detect squeeze (low volatility)
            avg_bandwidth = bandwidth.rolling(50).mean().iloc[-1]
            is_squeeze = current_bandwidth < avg_bandwidth * 0.75
            
            # Generate signals
            if current_price <= current_lower and prev_price > prev_lower:
                # Price touched/crossed below lower band
                action = "BUY"
                confidence = min(0.85, 0.6 + (1 - band_position) * 0.4)
                reasoning = f"Price touched lower Bollinger Band (oversold)"
                
            elif current_price >= current_upper and prev_price < prev_upper:
                # Price touched/crossed above upper band
                action = "SELL"
                confidence = min(0.85, 0.6 + band_position * 0.4)
                reasoning = f"Price touched upper Bollinger Band (overbought)"
                
            elif is_squeeze:
                # Bollinger squeeze - potential breakout coming
                action = "HOLD"
                confidence = 0.5
                reasoning = f"Bollinger Squeeze detected - awaiting breakout"
                
            elif band_position < 0.2:
                # Near lower band
                action = "BUY"
                confidence = 0.6
                reasoning = f"Price near lower band - potential bounce"
                
            elif band_position > 0.8:
                # Near upper band
                action = "SELL"
                confidence = 0.6
                reasoning = f"Price near upper band - potential reversal"
                
            else:
                # Price in middle of bands
                action = "HOLD"
                confidence = 0.3
                reasoning = f"Price within bands - no clear signal"
            
            # Volume confirmation
            volume_ratio = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
            if action != "HOLD" and volume_ratio > 1.5:
                confidence = min(0.95, confidence + 0.1)
                reasoning += f" (confirmed by {volume_ratio:.1f}x volume)"
            
            return {
                "action": action,
                "confidence": float(confidence),
                "metadata": {
                    "agent": self.name,
                    "symbol": symbol,
                    "reasoning": reasoning,
                    "timestamp": datetime.now().isoformat(),
                    "indicators": {
                        "upper_band": float(current_upper),
                        "middle_band": float(current_middle),
                        "lower_band": float(current_lower),
                        "price": float(current_price),
                        "band_position": float(band_position),
                        "bandwidth": float(current_bandwidth),
                        "is_squeeze": is_squeeze,
                        "volume_ratio": float(volume_ratio)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Bollinger Bands agent for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {"error": str(e), "agent": self.name}
            } 