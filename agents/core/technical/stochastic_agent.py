"""
Stochastic Oscillator Agent
Momentum indicator comparing closing price to price range
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class StochasticAgent:
    """Stochastic Oscillator momentum agent"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
        self.name = "stochastic_agent"
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.oversold = 20
        self.overbought = 80
        
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple:
        """Calculate %K and %D lines"""
        # Calculate raw %K
        lowest_low = low.rolling(window=self.k_period).min()
        highest_high = high.rolling(window=self.k_period).max()
        
        # Raw %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        raw_k = raw_k.fillna(50)  # Fill NaN with neutral value
        
        # Smooth %K (slow stochastic)
        k_percent = raw_k.rolling(window=self.smooth_k).mean()
        
        # %D is SMA of %K
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        return k_percent, d_percent
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal based on Stochastic Oscillator"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2mo")
            
            if data.empty or len(data) < self.k_period + self.d_period:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data", "agent": self.name}
                }
            
            # Calculate Stochastic
            k_percent, d_percent = self.calculate_stochastic(
                data['High'], data['Low'], data['Close']
            )
            
            # Current and previous values
            current_k = k_percent.iloc[-1]
            current_d = d_percent.iloc[-1]
            prev_k = k_percent.iloc[-2]
            prev_d = d_percent.iloc[-2]
            
            # Price data
            current_price = data['Close'].iloc[-1]
            price_change = (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
            
            # Detect crossovers
            k_crosses_above_d = current_k > current_d and prev_k <= prev_d
            k_crosses_below_d = current_k < current_d and prev_k >= prev_d
            
            # Check for divergences
            recent_prices = data['Close'].tail(10)
            recent_k = k_percent.tail(10)
            
            # Bullish divergence: price making lower lows, stochastic making higher lows
            bullish_divergence = (
                recent_prices.iloc[-1] < recent_prices.iloc[0] and
                recent_k.iloc[-1] > recent_k.iloc[0] and
                current_k < 30
            )
            
            # Bearish divergence: price making higher highs, stochastic making lower highs
            bearish_divergence = (
                recent_prices.iloc[-1] > recent_prices.iloc[0] and
                recent_k.iloc[-1] < recent_k.iloc[0] and
                current_k > 70
            )
            
            # Generate signals
            if k_crosses_above_d and current_k < self.oversold:
                # Bullish crossover in oversold zone
                action = "BUY"
                confidence = 0.85
                reasoning = f"Stochastic bullish crossover in oversold zone (%K={current_k:.1f})"
                
            elif k_crosses_below_d and current_k > self.overbought:
                # Bearish crossover in overbought zone
                action = "SELL"
                confidence = 0.85
                reasoning = f"Stochastic bearish crossover in overbought zone (%K={current_k:.1f})"
                
            elif bullish_divergence:
                # Bullish divergence
                action = "BUY"
                confidence = 0.75
                reasoning = f"Bullish divergence detected - price down, stochastic up"
                
            elif bearish_divergence:
                # Bearish divergence
                action = "SELL"
                confidence = 0.75
                reasoning = f"Bearish divergence detected - price up, stochastic down"
                
            elif current_k < self.oversold:
                # Oversold condition
                action = "BUY"
                confidence = 0.65
                reasoning = f"Stochastic oversold (%K={current_k:.1f})"
                
            elif current_k > self.overbought:
                # Overbought condition
                action = "SELL"
                confidence = 0.65
                reasoning = f"Stochastic overbought (%K={current_k:.1f})"
                
            else:
                # Neutral zone
                action = "HOLD"
                confidence = 0.3
                reasoning = f"Stochastic in neutral zone (%K={current_k:.1f})"
            
            # Adjust confidence based on %K-%D spread
            k_d_spread = abs(current_k - current_d)
            if action != "HOLD" and k_d_spread > 10:
                confidence = min(0.95, confidence + 0.1)
                reasoning += f" (strong momentum, spread={k_d_spread:.1f})"
            
            return {
                "action": action,
                "confidence": float(confidence),
                "metadata": {
                    "agent": self.name,
                    "symbol": symbol,
                    "reasoning": reasoning,
                    "timestamp": datetime.now().isoformat(),
                    "indicators": {
                        "k_percent": float(current_k),
                        "d_percent": float(current_d),
                        "k_d_spread": float(k_d_spread),
                        "is_oversold": current_k < self.oversold,
                        "is_overbought": current_k > self.overbought,
                        "bullish_divergence": bullish_divergence,
                        "bearish_divergence": bearish_divergence,
                        "price": float(current_price)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Stochastic agent for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {"error": str(e), "agent": self.name}
            } 