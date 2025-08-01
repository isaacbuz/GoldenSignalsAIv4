"""
EMA (Exponential Moving Average) Agent
Trend following using multiple EMAs and EMA ribbon
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class EMAAgent:
    """Exponential Moving Average trend following agent"""

    def __init__(self, ema_periods: List[int] = None):
        self.name = "ema_agent"
        # Default EMA periods for ribbon: 8, 13, 21, 34, 55, 89
        self.ema_periods = ema_periods or [8, 13, 21, 34, 55, 89]

    def calculate_ema_ribbon(self, prices: pd.Series) -> Dict[int, pd.Series]:
        """Calculate multiple EMAs forming a ribbon"""
        emas = {}
        for period in self.ema_periods:
            emas[period] = prices.ewm(span=period, adjust=False).mean()
        return emas

    def calculate_ribbon_width(self, emas: Dict[int, pd.Series]) -> pd.Series:
        """Calculate the width of the EMA ribbon (trend strength indicator)"""
        sorted_periods = sorted(self.ema_periods)
        fastest_ema = emas[sorted_periods[0]]
        slowest_ema = emas[sorted_periods[-1]]
        return (fastest_ema - slowest_ema) / slowest_ema * 100

    def check_ribbon_alignment(self, emas: Dict[int, pd.Series], index: int) -> str:
        """Check if EMAs are aligned (all bullish or bearish)"""
        values = []
        for period in sorted(self.ema_periods):
            values.append(emas[period].iloc[index])

        # Check if values are in ascending order (bearish) or descending order (bullish)
        is_bullish = all(values[i] > values[i+1] for i in range(len(values)-1))
        is_bearish = all(values[i] < values[i+1] for i in range(len(values)-1))

        if is_bullish:
            return "bullish"
        elif is_bearish:
            return "bearish"
        else:
            return "mixed"

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal based on EMA ribbon"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo")

            if data.empty or len(data) < max(self.ema_periods):
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data", "agent": self.name}
                }

            # Calculate EMA ribbon
            emas = self.calculate_ema_ribbon(data['Close'])
            ribbon_width = self.calculate_ribbon_width(emas)

            # Current values
            current_price = data['Close'].iloc[-1]
            current_alignment = self.check_ribbon_alignment(emas, -1)
            prev_alignment = self.check_ribbon_alignment(emas, -2)
            current_width = ribbon_width.iloc[-1]

            # Fast and slow EMA values
            fast_ema = emas[self.ema_periods[0]].iloc[-1]
            slow_ema = emas[self.ema_periods[-1]].iloc[-1]

            # Price position relative to ribbon
            above_all_emas = all(current_price > emas[period].iloc[-1] for period in self.ema_periods)
            below_all_emas = all(current_price < emas[period].iloc[-1] for period in self.ema_periods)

            # EMA crossovers
            fast_crosses_slow_up = (
                emas[self.ema_periods[0]].iloc[-1] > emas[self.ema_periods[-1]].iloc[-1] and
                emas[self.ema_periods[0]].iloc[-2] <= emas[self.ema_periods[-1]].iloc[-2]
            )
            fast_crosses_slow_down = (
                emas[self.ema_periods[0]].iloc[-1] < emas[self.ema_periods[-1]].iloc[-1] and
                emas[self.ema_periods[0]].iloc[-2] >= emas[self.ema_periods[-1]].iloc[-2]
            )

            # Generate signals
            if current_alignment == "bullish" and prev_alignment != "bullish":
                # Ribbon turned bullish
                action = "BUY"
                confidence = 0.85
                reasoning = "EMA ribbon aligned bullish - strong uptrend starting"

            elif current_alignment == "bearish" and prev_alignment != "bearish":
                # Ribbon turned bearish
                action = "SELL"
                confidence = 0.85
                reasoning = "EMA ribbon aligned bearish - strong downtrend starting"

            elif fast_crosses_slow_up and current_alignment != "bearish":
                # Fast EMA crossed above slow EMA
                action = "BUY"
                confidence = 0.75
                reasoning = f"Fast EMA ({self.ema_periods[0]}) crossed above slow EMA ({self.ema_periods[-1]})"

            elif fast_crosses_slow_down and current_alignment != "bullish":
                # Fast EMA crossed below slow EMA
                action = "SELL"
                confidence = 0.75
                reasoning = f"Fast EMA ({self.ema_periods[0]}) crossed below slow EMA ({self.ema_periods[-1]})"

            elif above_all_emas and current_alignment == "bullish":
                # Price above all EMAs in bullish ribbon
                action = "BUY"
                confidence = 0.7
                reasoning = "Price above all EMAs in bullish trend"

            elif below_all_emas and current_alignment == "bearish":
                # Price below all EMAs in bearish ribbon
                action = "SELL"
                confidence = 0.7
                reasoning = "Price below all EMAs in bearish trend"

            elif current_alignment == "mixed":
                # EMAs are mixed - no clear trend
                action = "HOLD"
                confidence = 0.4
                reasoning = "EMA ribbon mixed - no clear trend"

            else:
                # Default hold
                action = "HOLD"
                confidence = 0.3
                reasoning = f"EMA ribbon {current_alignment} but no trigger"

            # Adjust confidence based on ribbon width (trend strength)
            avg_width = ribbon_width.rolling(20).mean().iloc[-1]
            if abs(current_width) > abs(avg_width) * 1.5:
                confidence = min(0.95, confidence + 0.1)
                reasoning += f" (strong trend, width={current_width:.2f}%)"

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
                        "fast_ema": float(fast_ema),
                        "slow_ema": float(slow_ema),
                        "ribbon_alignment": current_alignment,
                        "ribbon_width": float(current_width),
                        "above_all_emas": above_all_emas,
                        "below_all_emas": below_all_emas,
                        "ema_values": {str(p): float(emas[p].iloc[-1]) for p in self.ema_periods}
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error in EMA agent for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {"error": str(e), "agent": self.name}
            }
