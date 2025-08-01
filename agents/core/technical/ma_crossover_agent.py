"""
Moving Average Crossover Agent
Generates signals based on MA crossovers (Golden Cross/Death Cross)
"""

import logging
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class MACrossoverAgent:
    """Moving Average crossover trading agent"""

    def __init__(self, fast_ma: int = 50, slow_ma: int = 200):
        self.name = "ma_crossover_agent"
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate signal based on MA crossovers"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")

            if data.empty or len(data) < self.slow_ma:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data", "agent": self.name}
                }

            # Calculate moving averages
            fast_sma = data['Close'].rolling(self.fast_ma).mean()
            slow_sma = data['Close'].rolling(self.slow_ma).mean()

            # Current values
            current_fast = fast_sma.iloc[-1]
            current_slow = slow_sma.iloc[-1]
            prev_fast = fast_sma.iloc[-2]
            prev_slow = slow_sma.iloc[-2]
            current_price = data['Close'].iloc[-1]

            # Detect crossovers
            if current_fast > current_slow and prev_fast <= prev_slow:
                # Golden Cross
                action = "BUY"
                confidence = 0.8
                reasoning = f"Golden Cross: {self.fast_ma}MA crossed above {self.slow_ma}MA"
            elif current_fast < current_slow and prev_fast >= prev_slow:
                # Death Cross
                action = "SELL"
                confidence = 0.8
                reasoning = f"Death Cross: {self.fast_ma}MA crossed below {self.slow_ma}MA"
            elif current_fast > current_slow:
                # Bullish trend
                action = "HOLD"
                confidence = 0.4
                reasoning = f"Bullish trend: {self.fast_ma}MA above {self.slow_ma}MA"
            else:
                # Bearish trend
                action = "HOLD"
                confidence = 0.4
                reasoning = f"Bearish trend: {self.fast_ma}MA below {self.slow_ma}MA"

            # Adjust confidence based on separation
            separation = abs(current_fast - current_slow) / current_slow * 100
            if action in ["BUY", "SELL"]:
                confidence = min(0.95, confidence + separation * 0.01)

            return {
                "action": action,
                "confidence": float(confidence),
                "metadata": {
                    "agent": self.name,
                    "symbol": symbol,
                    "reasoning": reasoning,
                    "timestamp": datetime.now().isoformat(),
                    "indicators": {
                        f"ma_{self.fast_ma}": float(current_fast),
                        f"ma_{self.slow_ma}": float(current_slow),
                        "ma_separation_pct": float(separation),
                        "price": float(current_price),
                        "price_to_fast_ma": float((current_price - current_fast) / current_fast * 100)
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error in MA Crossover agent for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {"error": str(e), "agent": self.name}
            }
