"""
ATR (Average True Range) Agent
Volatility measurement and dynamic stop loss/target generation
"""

import logging
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class ATRAgent:
    """Average True Range volatility and risk management agent"""

    def __init__(self, atr_period: int = 14, risk_multiplier: float = 2.0):
        self.name = "atr_agent"
        self.atr_period = atr_period
        self.risk_multiplier = risk_multiplier

    def calculate_true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range"""
        # True Range = max of:
        # 1. Current High - Current Low
        # 2. abs(Current High - Previous Close)
        # 3. abs(Current Low - Previous Close)

        high_low = high - low
        high_close = abs(high - close.shift(1))
        low_close = abs(low - close.shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Average True Range"""
        true_range = self.calculate_true_range(high, low, close)
        atr = true_range.rolling(window=self.atr_period).mean()
        return atr

    def calculate_volatility_regime(self, atr: pd.Series) -> str:
        """Determine current volatility regime"""
        current_atr = atr.iloc[-1]
        recent_avg = atr.tail(50).mean()
        long_avg = atr.tail(200).mean() if len(atr) >= 200 else recent_avg

        if current_atr > recent_avg * 1.5:
            return "high"
        elif current_atr < recent_avg * 0.7:
            return "low"
        else:
            return "normal"

    def calculate_stops_and_targets(self, price: float, atr: float, trend: str) -> Tuple[float, float, float]:
        """Calculate dynamic stop loss and profit targets based on ATR"""
        if trend == "bullish":
            stop_loss = price - (atr * self.risk_multiplier)
            target_1 = price + (atr * 1.5)
            target_2 = price + (atr * 3.0)
        elif trend == "bearish":
            stop_loss = price + (atr * self.risk_multiplier)
            target_1 = price - (atr * 1.5)
            target_2 = price - (atr * 3.0)
        else:
            # Neutral - wider stops
            stop_loss = price - (atr * self.risk_multiplier * 1.5)
            target_1 = price + (atr * 2.0)
            target_2 = price + (atr * 4.0)

        return stop_loss, target_1, target_2

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal based on ATR volatility analysis"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo")

            if data.empty or len(data) < self.atr_period + 1:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data", "agent": self.name}
                }

            # Calculate ATR
            atr_series = self.calculate_atr(data['High'], data['Low'], data['Close'])
            current_atr = atr_series.iloc[-1]
            prev_atr = atr_series.iloc[-2]

            # Price and trend analysis
            current_price = data['Close'].iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]

            # Determine trend
            if current_price > sma_20 > sma_50:
                trend = "bullish"
            elif current_price < sma_20 < sma_50:
                trend = "bearish"
            else:
                trend = "neutral"

            # Volatility analysis
            volatility_regime = self.calculate_volatility_regime(atr_series)
            atr_change = (current_atr - prev_atr) / prev_atr

            # ATR as percentage of price
            atr_percent = (current_atr / current_price) * 100

            # Calculate stops and targets
            stop_loss, target_1, target_2 = self.calculate_stops_and_targets(
                current_price, current_atr, trend
            )

            # Volatility expansion/contraction
            recent_atr_avg = atr_series.tail(20).mean()
            volatility_expanding = current_atr > recent_atr_avg * 1.2
            volatility_contracting = current_atr < recent_atr_avg * 0.8

            # Generate signals based on volatility patterns
            if volatility_expanding and trend == "bullish":
                # Volatility expansion in uptrend - potential breakout
                action = "BUY"
                confidence = 0.75
                reasoning = "Volatility expansion in uptrend - potential breakout"

            elif volatility_expanding and trend == "bearish":
                # Volatility expansion in downtrend - potential breakdown
                action = "SELL"
                confidence = 0.75
                reasoning = "Volatility expansion in downtrend - potential breakdown"

            elif volatility_contracting and volatility_regime == "low":
                # Low volatility - potential breakout setup
                action = "HOLD"
                confidence = 0.6
                reasoning = "Low volatility squeeze - awaiting breakout direction"

            elif atr_percent > 5:
                # Very high volatility - reduce position or avoid
                action = "HOLD"
                confidence = 0.7
                reasoning = f"High volatility environment (ATR {atr_percent:.1f}% of price) - risk management"

            else:
                # Normal conditions - follow trend with ATR-based stops
                if trend == "bullish":
                    action = "BUY"
                    confidence = 0.6
                    reasoning = f"Bullish trend with normal volatility"
                elif trend == "bearish":
                    action = "SELL"
                    confidence = 0.6
                    reasoning = f"Bearish trend with normal volatility"
                else:
                    action = "HOLD"
                    confidence = 0.4
                    reasoning = "Neutral trend - no clear direction"

            # Adjust confidence based on volatility regime
            if volatility_regime == "high" and action != "HOLD":
                confidence *= 0.8  # Reduce confidence in high volatility
                reasoning += " (adjusted for high volatility)"

            return {
                "action": action,
                "confidence": float(confidence),
                "metadata": {
                    "agent": self.name,
                    "symbol": symbol,
                    "reasoning": reasoning,
                    "timestamp": datetime.now().isoformat(),
                    "indicators": {
                        "atr": float(current_atr),
                        "atr_percent": float(atr_percent),
                        "volatility_regime": volatility_regime,
                        "volatility_expanding": volatility_expanding,
                        "volatility_contracting": volatility_contracting,
                        "trend": trend,
                        "price": float(current_price),
                        "stop_loss": float(stop_loss),
                        "target_1": float(target_1),
                        "target_2": float(target_2),
                        "risk_reward_1": float(abs(target_1 - current_price) / abs(stop_loss - current_price)),
                        "risk_reward_2": float(abs(target_2 - current_price) / abs(stop_loss - current_price))
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error in ATR agent for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {"error": str(e), "agent": self.name}
            }
