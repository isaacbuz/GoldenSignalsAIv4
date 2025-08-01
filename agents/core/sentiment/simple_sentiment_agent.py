"""
Sentiment Analysis Agent
Analyzes market sentiment using technical indicators as sentiment proxies
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class SimpleSentimentAgent:
    """
    Sentiment Analysis trading agent

    Signals:
    - Fear/Greed indicators
    - Market breadth analysis
    - Put/Call ratio proxy
    - Volatility sentiment
    """

    def __init__(self):
        self.name = "SimpleSentimentAgent"
        self.fear_greed_components = 5  # Number of sentiment components

    def calculate_rsi_sentiment(self, df: pd.DataFrame, period: int = 14) -> float:
        """RSI as sentiment indicator (oversold = fear, overbought = greed)"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # Convert to sentiment score (0 = extreme fear, 100 = extreme greed)
        if current_rsi < 30:
            return current_rsi * 1.5  # Scale up fear
        elif current_rsi > 70:
            return 70 + (current_rsi - 70) * 1.5  # Scale up greed
        else:
            return current_rsi

    def calculate_volume_sentiment(self, df: pd.DataFrame) -> float:
        """Volume patterns as sentiment (high volume on up days = greed)"""
        recent = df.tail(20)

        up_volume = recent[recent['Close'] > recent['Open']]['Volume'].sum()
        down_volume = recent[recent['Close'] < recent['Open']]['Volume'].sum()
        total_volume = up_volume + down_volume

        if total_volume == 0:
            return 50

        # More volume on up days = higher sentiment
        volume_sentiment = (up_volume / total_volume) * 100
        return volume_sentiment

    def calculate_momentum_sentiment(self, df: pd.DataFrame) -> float:
        """Price momentum as sentiment indicator"""
        # Short-term vs long-term performance
        returns_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
        returns_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100

        # Strong short-term momentum = greed
        momentum_score = 50 + (returns_5d * 2) + (returns_20d * 0.5)

        # Cap between 0 and 100
        return max(0, min(100, momentum_score))

    def calculate_volatility_sentiment(self, df: pd.DataFrame) -> float:
        """Volatility as fear indicator (high vol = fear)"""
        # Calculate recent volatility
        returns = df['Close'].pct_change()
        recent_vol = returns.tail(10).std() * np.sqrt(252)  # Annualized
        historical_vol = returns.tail(60).std() * np.sqrt(252)

        if historical_vol == 0:
            return 50

        vol_ratio = recent_vol / historical_vol

        # High volatility = fear (lower score)
        if vol_ratio > 1.5:
            return 20  # High fear
        elif vol_ratio > 1.2:
            return 35
        elif vol_ratio < 0.8:
            return 80  # Low volatility = complacency/greed
        else:
            return 50

    def calculate_breadth_sentiment(self, df: pd.DataFrame) -> float:
        """Market breadth using price position"""
        # How many recent days closed above their midpoint
        recent = df.tail(20)

        above_mid = 0
        for _, row in recent.iterrows():
            mid = (row['High'] + row['Low']) / 2
            if row['Close'] > mid:
                above_mid += 1

        # More days closing above mid = bullish sentiment
        breadth_score = (above_mid / len(recent)) * 100
        return breadth_score

    def calculate_fear_greed_index(self, components: Dict[str, float]) -> float:
        """Combine components into overall fear/greed index"""
        # Equal weighting for simplicity
        scores = list(components.values())
        return np.mean(scores)

    def interpret_sentiment(self, score: float) -> str:
        """Interpret sentiment score"""
        if score < 20:
            return "Extreme Fear"
        elif score < 40:
            return "Fear"
        elif score < 60:
            return "Neutral"
        elif score < 80:
            return "Greed"
        else:
            return "Extreme Greed"

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate Sentiment trading signal"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")

            if df.empty or len(df) < 60:
                return self._create_signal(symbol, "NEUTRAL", 0, "Insufficient data")

            # Calculate sentiment components
            components = {
                'rsi_sentiment': self.calculate_rsi_sentiment(df),
                'volume_sentiment': self.calculate_volume_sentiment(df),
                'momentum_sentiment': self.calculate_momentum_sentiment(df),
                'volatility_sentiment': self.calculate_volatility_sentiment(df),
                'breadth_sentiment': self.calculate_breadth_sentiment(df)
            }

            # Overall sentiment score
            sentiment_score = self.calculate_fear_greed_index(components)
            sentiment_label = self.interpret_sentiment(sentiment_score)

            # Get current price info
            current_price = df['Close'].iloc[-1]
            sma_50 = df['Close'].tail(50).mean()

            signals = []
            strength = 0

            # 1. Overall Sentiment Signal
            signals.append(f"{sentiment_label} ({sentiment_score:.0f}/100)")

            if sentiment_score < 25:  # Extreme Fear
                signals.append("Contrarian buy opportunity")
                strength += 0.5
            elif sentiment_score < 40:  # Fear
                signals.append("Market fearful - potential opportunity")
                strength += 0.2
            elif sentiment_score > 75:  # Extreme Greed
                signals.append("Contrarian sell signal")
                strength -= 0.5
            elif sentiment_score > 60:  # Greed
                signals.append("Market greedy - caution advised")
                strength -= 0.2
            else:
                signals.append("Neutral sentiment")

            # 2. Component Analysis
            # RSI Sentiment
            if components['rsi_sentiment'] < 30:
                signals.append("RSI showing oversold fear")
                strength += 0.2
            elif components['rsi_sentiment'] > 70:
                signals.append("RSI showing overbought greed")
                strength -= 0.2

            # Volume Sentiment
            if components['volume_sentiment'] > 70:
                signals.append("Strong buying volume sentiment")
                if sentiment_score < 40:  # Bullish divergence
                    strength += 0.2
            elif components['volume_sentiment'] < 30:
                signals.append("Strong selling volume sentiment")
                if sentiment_score > 60:  # Bearish divergence
                    strength -= 0.2

            # Volatility Sentiment
            if components['volatility_sentiment'] < 30:
                signals.append("High volatility fear")
                if current_price < sma_50:
                    strength += 0.1  # Buy fear below MA
            elif components['volatility_sentiment'] > 70:
                signals.append("Low volatility complacency")
                strength *= 0.9  # Reduce position size

            # 3. Sentiment Extremes
            extreme_components = sum(1 for v in components.values() if v < 20 or v > 80)
            if extreme_components >= 3:
                signals.append(f"{extreme_components} components at extremes")
                # Multiple extremes often mark turning points
                if sentiment_score < 40:
                    strength += 0.2
                else:
                    strength -= 0.2

            # 4. Sentiment Trend
            # Simple trend using momentum component
            if components['momentum_sentiment'] > 60 and sentiment_score < 70:
                signals.append("Positive momentum, reasonable sentiment")
                strength += 0.1
            elif components['momentum_sentiment'] < 40 and sentiment_score > 30:
                signals.append("Negative momentum, not oversold")
                strength -= 0.1

            # 5. Smart Money Indicator (simplified)
            # Low volatility + steady volume = smart accumulation
            if components['volatility_sentiment'] > 60 and components['volume_sentiment'] > 45 and components['volume_sentiment'] < 55:
                signals.append("Possible smart money accumulation")
                strength += 0.1

            # Determine action
            if strength >= 0.3:
                action = "BUY"
            elif strength <= -0.3:
                action = "SELL"
            else:
                action = "NEUTRAL"

            confidence = min(abs(strength), 1.0)

            return self._create_signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                reason=f"Sentiment: {', '.join(signals)}",
                data={
                    "price": float(current_price),
                    "sentiment_score": float(sentiment_score),
                    "sentiment_label": sentiment_label,
                    "components": {k: float(v) for k, v in components.items()},
                    "extreme_count": extreme_components,
                    "signals": signals
                }
            )

        except Exception as e:
            logger.error(f"Error generating Sentiment signal for {symbol}: {str(e)}")
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
