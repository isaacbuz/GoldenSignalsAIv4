"""
Fibonacci Retracement Agent
Identifies key support and resistance levels based on Fibonacci ratios
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class FibonacciAgent:
    """
    Fibonacci Retracement trading agent

    Signals:
    - Price near key Fibonacci levels
    - Bounce/break of levels
    - Confluence with other indicators
    """

    def __init__(self):
        self.name = "FibonacciAgent"
        # Fibonacci levels
        self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.lookback_period = 50  # Days to find swing high/low
        self.proximity_threshold = 0.005  # 0.5% proximity to level

    def find_swing_points(self, df: pd.DataFrame) -> Tuple[float, float, int, int]:
        """Find recent swing high and low points"""
        # Get recent data
        recent = df.tail(self.lookback_period)

        # Find swing high and low
        swing_high = recent['High'].max()
        swing_low = recent['Low'].min()

        # Find indexes
        high_idx = recent['High'].idxmax()
        low_idx = recent['Low'].idxmin()

        return swing_high, swing_low, high_idx, low_idx

    def calculate_fib_levels(self, high: float, low: float, is_uptrend: bool) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels"""
        levels = {}
        diff = high - low

        for level in self.fib_levels:
            if is_uptrend:
                # In uptrend, we retrace from high
                levels[level] = high - (diff * level)
            else:
                # In downtrend, we retrace from low
                levels[level] = low + (diff * level)

        return levels

    def check_level_proximity(self, price: float, level: float) -> bool:
        """Check if price is near a Fibonacci level"""
        return abs(price - level) / level <= self.proximity_threshold

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate Fibonacci trading signal"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")

            if df.empty or len(df) < self.lookback_period:
                return self._create_signal(symbol, "NEUTRAL", 0, "Insufficient data")

            # Find swing points
            swing_high, swing_low, high_idx, low_idx = self.find_swing_points(df)

            # Determine trend direction
            is_uptrend = high_idx > low_idx

            # Calculate Fibonacci levels
            fib_levels = self.calculate_fib_levels(swing_high, swing_low, is_uptrend)

            # Get current price
            current_price = df['Close'].iloc[-1]

            # Check proximity to levels
            signals = []
            strength = 0
            nearest_level = None
            nearest_distance = float('inf')

            for fib_ratio, level_price in fib_levels.items():
                distance = abs(current_price - level_price)

                # Track nearest level
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_level = (fib_ratio, level_price)

                # Check if price is near this level
                if self.check_level_proximity(current_price, level_price):
                    signals.append(f"Near {fib_ratio:.1%} level at ${level_price:.2f}")

                    # Key levels have stronger signals
                    if fib_ratio in [0.382, 0.5, 0.618]:
                        # Check if bouncing off level
                        recent_low = df['Low'].tail(3).min()
                        recent_high = df['High'].tail(3).max()

                        if is_uptrend:
                            if recent_low > level_price * 0.995 and current_price > level_price:
                                signals.append(f"Bouncing off {fib_ratio:.1%}")
                                strength += 0.4
                            elif current_price < level_price:
                                signals.append(f"Broke below {fib_ratio:.1%}")
                                strength -= 0.3
                        else:
                            if recent_high < level_price * 1.005 and current_price < level_price:
                                signals.append(f"Rejecting from {fib_ratio:.1%}")
                                strength -= 0.4
                            elif current_price > level_price:
                                signals.append(f"Broke above {fib_ratio:.1%}")
                                strength += 0.3

            # Price momentum
            price_change = (current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5]

            if is_uptrend:
                if price_change > 0.01:
                    signals.append("Upward momentum")
                    strength += 0.2
                elif price_change < -0.01:
                    signals.append("Pullback in uptrend")
                    strength -= 0.1
            else:
                if price_change < -0.01:
                    signals.append("Downward momentum")
                    strength -= 0.2
                elif price_change > 0.01:
                    signals.append("Bounce in downtrend")
                    strength += 0.1

            # Volume confirmation
            avg_volume = df['Volume'].tail(20).mean()
            current_volume = df['Volume'].iloc[-1]
            if current_volume > avg_volume * 1.5:
                signals.append("High volume")
                strength = strength * 1.2  # Amplify signal

            # Determine action
            if strength >= 0.3:
                action = "BUY"
            elif strength <= -0.3:
                action = "SELL"
            else:
                action = "NEUTRAL"

            confidence = min(abs(strength), 1.0)

            # Create detailed reason
            trend_str = "uptrend" if is_uptrend else "downtrend"
            reason = f"Fibonacci ({trend_str}): {', '.join(signals)}"

            return self._create_signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                reason=reason,
                data={
                    "price": float(current_price),
                    "swing_high": float(swing_high),
                    "swing_low": float(swing_low),
                    "is_uptrend": is_uptrend,
                    "fib_levels": {f"{k:.1%}": float(v) for k, v in fib_levels.items()},
                    "nearest_level": {
                        "ratio": f"{nearest_level[0]:.1%}" if nearest_level else None,
                        "price": float(nearest_level[1]) if nearest_level else None
                    },
                    "signals": signals
                }
            )

        except Exception as e:
            logger.error(f"Error generating Fibonacci signal for {symbol}: {str(e)}")
            return self._create_signal(symbol, "ERROR", 0, str(e))

    def _create_signal(self, symbol: str, action: str, confidence: float,
                      reason: str, data: Dict = None) -> Dict[str, Any]:
        """Create standardized signal output"""
        return {
            "agent": self.name,
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
