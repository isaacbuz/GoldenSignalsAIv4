"""
Volume Spike Agent
Detects unusual volume patterns that often precede price movements
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VolumeSpikeAgent:
    """Agent that detects volume spikes and unusual trading activity"""
    
    def __init__(self, spike_threshold: float = 2.0, lookback_period: int = 20):
        self.name = "volume_spike_agent"
        self.spike_threshold = spike_threshold
        self.lookback_period = lookback_period
        
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate signal based on volume analysis"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2mo")
            
            if data.empty or len(data) < self.lookback_period:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data", "agent": self.name}
                }
            
            # Calculate volume metrics
            volume_sma = data['Volume'].rolling(self.lookback_period).mean()
            volume_std = data['Volume'].rolling(self.lookback_period).std()
            current_volume = data['Volume'].iloc[-1]
            avg_volume = volume_sma.iloc[-1]
            
            # Volume spike detection
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            z_score = (current_volume - avg_volume) / volume_std.iloc[-1] if volume_std.iloc[-1] > 0 else 0
            
            # Price action
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
            
            # Determine signal
            if volume_ratio > self.spike_threshold:
                if price_change > 0.01:  # 1% up with high volume
                    action = "BUY"
                    confidence = min(0.9, 0.5 + (volume_ratio - self.spike_threshold) * 0.2)
                    reasoning = f"High volume ({volume_ratio:.1f}x avg) with positive price action"
                elif price_change < -0.01:  # 1% down with high volume
                    action = "SELL"
                    confidence = min(0.9, 0.5 + (volume_ratio - self.spike_threshold) * 0.2)
                    reasoning = f"High volume ({volume_ratio:.1f}x avg) with negative price action"
                else:
                    action = "HOLD"
                    confidence = 0.4
                    reasoning = f"High volume spike but unclear price direction"
            else:
                action = "HOLD"
                confidence = 0.2
                reasoning = "Normal volume levels"
            
            return {
                "action": action,
                "confidence": float(confidence),
                "metadata": {
                    "agent": self.name,
                    "symbol": symbol,
                    "reasoning": reasoning,
                    "timestamp": datetime.now().isoformat(),
                    "indicators": {
                        "volume_ratio": float(volume_ratio),
                        "volume_z_score": float(z_score),
                        "current_volume": int(current_volume),
                        "avg_volume": int(avg_volume),
                        "price_change_pct": float(price_change * 100),
                        "price": float(data['Close'].iloc[-1])
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Volume Spike agent for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {"error": str(e), "agent": self.name}
            } 