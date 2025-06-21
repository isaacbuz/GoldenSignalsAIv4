"""
VWAP (Volume Weighted Average Price) Agent
Institutional-level trading indicator for intraday price levels
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)

class VWAPAgent:
    """VWAP institutional trading agent"""
    
    def __init__(self):
        self.name = "vwap_agent"
        
    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP for the trading session"""
        # VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap
    
    def calculate_vwap_bands(self, data: pd.DataFrame, vwap: pd.Series, num_std: float = 2.0) -> tuple:
        """Calculate VWAP bands based on standard deviation"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Calculate squared deviations weighted by volume
        squared_diff = (typical_price - vwap) ** 2
        variance = (squared_diff * data['Volume']).cumsum() / data['Volume'].cumsum()
        std_dev = np.sqrt(variance)
        
        upper_band = vwap + (std_dev * num_std)
        lower_band = vwap - (std_dev * num_std)
        
        return upper_band, lower_band
    
    def is_market_hours(self) -> bool:
        """Check if current time is during market hours (9:30 AM - 4:00 PM EST)"""
        now = datetime.now().time()
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= now <= market_close
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal based on VWAP"""
        try:
            # For VWAP, we need intraday data
            ticker = yf.Ticker(symbol)
            
            # Get intraday data (1-minute intervals for current day)
            intraday_data = ticker.history(period="1d", interval="1m")
            
            # If no intraday data, fall back to daily data
            if intraday_data.empty or len(intraday_data) < 30:
                # Use daily data as fallback
                daily_data = ticker.history(period="5d", interval="1h")
                if daily_data.empty:
                    return {
                        "action": "HOLD",
                        "confidence": 0.0,
                        "metadata": {"error": "Insufficient data", "agent": self.name}
                    }
                data = daily_data
            else:
                data = intraday_data
            
            # Calculate VWAP and bands
            vwap = self.calculate_vwap(data)
            upper_band, lower_band = self.calculate_vwap_bands(data, vwap)
            
            # Current values
            current_price = data['Close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            # Previous values
            prev_price = data['Close'].iloc[-2]
            prev_vwap = vwap.iloc[-2]
            
            # Calculate price position relative to VWAP
            price_to_vwap = ((current_price - current_vwap) / current_vwap) * 100
            
            # Volume analysis
            avg_volume = data['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Trend analysis
            vwap_slope = (current_vwap - vwap.iloc[-10]) / vwap.iloc[-10] * 100 if len(vwap) >= 10 else 0
            price_trend = "bullish" if current_price > current_vwap else "bearish"
            
            # VWAP crossover detection
            price_crosses_above_vwap = current_price > current_vwap and prev_price <= prev_vwap
            price_crosses_below_vwap = current_price < current_vwap and prev_price >= prev_vwap
            
            # Generate signals
            if price_crosses_above_vwap and volume_ratio > 1.2:
                # Price crossed above VWAP with volume
                action = "BUY"
                confidence = 0.8
                reasoning = "Price crossed above VWAP with strong volume - institutional buying"
                
            elif price_crosses_below_vwap and volume_ratio > 1.2:
                # Price crossed below VWAP with volume
                action = "SELL"
                confidence = 0.8
                reasoning = "Price crossed below VWAP with strong volume - institutional selling"
                
            elif current_price > current_upper:
                # Price above upper band - overbought
                action = "SELL"
                confidence = 0.7
                reasoning = f"Price {price_to_vwap:.1f}% above VWAP upper band - overbought"
                
            elif current_price < current_lower:
                # Price below lower band - oversold
                action = "BUY"
                confidence = 0.7
                reasoning = f"Price {abs(price_to_vwap):.1f}% below VWAP lower band - oversold"
                
            elif current_price > current_vwap and vwap_slope > 0:
                # Price above rising VWAP - bullish
                action = "BUY"
                confidence = 0.65
                reasoning = "Price above rising VWAP - bullish institutional flow"
                
            elif current_price < current_vwap and vwap_slope < 0:
                # Price below falling VWAP - bearish
                action = "SELL"
                confidence = 0.65
                reasoning = "Price below falling VWAP - bearish institutional flow"
                
            elif abs(price_to_vwap) < 0.5:
                # Price near VWAP - neutral
                action = "HOLD"
                confidence = 0.5
                reasoning = "Price near VWAP - equilibrium zone"
                
            else:
                # Default based on position
                if current_price > current_vwap:
                    action = "HOLD"
                    confidence = 0.4
                    reasoning = "Price above VWAP but no clear signal"
                else:
                    action = "HOLD"
                    confidence = 0.4
                    reasoning = "Price below VWAP but no clear signal"
            
            # Adjust confidence for market hours (if applicable)
            if hasattr(intraday_data, 'empty') and not intraday_data.empty:
                if not self.is_market_hours() and action != "HOLD":
                    confidence *= 0.7
                    reasoning += " (after-hours trading)"
            
            return {
                "action": action,
                "confidence": float(confidence),
                "metadata": {
                    "agent": self.name,
                    "symbol": symbol,
                    "reasoning": reasoning,
                    "timestamp": datetime.now().isoformat(),
                    "indicators": {
                        "price": float(current_price),
                        "vwap": float(current_vwap),
                        "vwap_upper": float(current_upper),
                        "vwap_lower": float(current_lower),
                        "price_to_vwap_pct": float(price_to_vwap),
                        "vwap_slope_pct": float(vwap_slope),
                        "volume_ratio": float(volume_ratio),
                        "price_position": price_trend,
                        "is_above_vwap": current_price > current_vwap,
                        "distance_to_vwap": float(current_price - current_vwap)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in VWAP agent for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {"error": str(e), "agent": self.name}
            } 