"""
MACD (Moving Average Convergence Divergence) Agent
Generates signals based on MACD crossovers
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MACDAgent:
    """MACD-based trading agent for trend following"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.name = "macd_agent"
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD, signal line, and histogram"""
        # Calculate EMAs
        ema_fast = prices.ewm(span=self.fast_period).mean()
        ema_slow = prices.ewm(span=self.slow_period).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        signal = macd.ewm(span=self.signal_period).mean()
        
        # Histogram
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal based on MACD"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            
            if data.empty or len(data) < self.slow_period + self.signal_period:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data", "agent": self.name}
                }
            
            # Calculate MACD
            macd, signal, histogram = self.calculate_macd(data['Close'])
            
            # Get current values
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            current_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2]
            
            # Determine signal
            if current_hist > 0 and prev_hist <= 0:
                # Bullish crossover
                action = "BUY"
                confidence = min(0.85, 0.6 + abs(current_hist) * 10)
                reasoning = f"MACD bullish crossover detected"
            elif current_hist < 0 and prev_hist >= 0:
                # Bearish crossover
                action = "SELL"
                confidence = min(0.85, 0.6 + abs(current_hist) * 10)
                reasoning = f"MACD bearish crossover detected"
            else:
                # No crossover
                action = "HOLD"
                confidence = 0.3
                reasoning = f"No MACD crossover signal"
            
            return {
                "action": action,
                "confidence": float(confidence),
                "metadata": {
                    "agent": self.name,
                    "symbol": symbol,
                    "reasoning": reasoning,
                    "timestamp": datetime.now().isoformat(),
                    "indicators": {
                        "macd": float(current_macd),
                        "signal": float(current_signal),
                        "histogram": float(current_hist),
                        "price": float(data['Close'].iloc[-1])
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in MACD agent for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {"error": str(e), "agent": self.name}
            } 