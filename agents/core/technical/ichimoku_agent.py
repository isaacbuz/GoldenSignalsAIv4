"""
Ichimoku Cloud Agent
Multi-timeframe trend analysis with cloud signals
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class IchimokuAgent:
    """
    Ichimoku Cloud trading agent

    Signals:
    - Price above/below cloud
    - Cloud color (bullish/bearish)
    - TK cross (Tenkan/Kijun)
    - Chikou span confirmation
    """

    def __init__(self):
        self.name = "IchimokuAgent"
        # Ichimoku parameters
        self.tenkan_period = 9      # Conversion line
        self.kijun_period = 26      # Base line
        self.senkou_span_b = 52     # Leading span B
        self.chikou_shift = 26      # Lagging span

    def calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line)
        high_9 = df['High'].rolling(window=self.tenkan_period).max()
        low_9 = df['Low'].rolling(window=self.tenkan_period).min()
        df['Tenkan'] = (high_9 + low_9) / 2

        # Kijun-sen (Base Line)
        high_26 = df['High'].rolling(window=self.kijun_period).max()
        low_26 = df['Low'].rolling(window=self.kijun_period).min()
        df['Kijun'] = (high_26 + low_26) / 2

        # Senkou Span A (Leading Span A) - shifted 26 periods ahead
        df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(self.kijun_period)

        # Senkou Span B (Leading Span B) - shifted 26 periods ahead
        high_52 = df['High'].rolling(window=self.senkou_span_b).max()
        low_52 = df['Low'].rolling(window=self.senkou_span_b).min()
        df['Senkou_B'] = ((high_52 + low_52) / 2).shift(self.kijun_period)

        # Chikou Span (Lagging Span) - shifted 26 periods back
        df['Chikou'] = df['Close'].shift(-self.chikou_shift)

        return df

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate Ichimoku trading signal"""
        try:
            # Fetch data (need extra for shifted indicators)
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo", interval="1d")

            if df.empty or len(df) < self.senkou_span_b + self.kijun_period:
                return self._create_signal(symbol, "NEUTRAL", 0, "Insufficient data")

            # Calculate Ichimoku
            df = self.calculate_ichimoku(df)

            # Get current values
            current_idx = -1
            price = df['Close'].iloc[current_idx]
            tenkan = df['Tenkan'].iloc[current_idx]
            kijun = df['Kijun'].iloc[current_idx]
            senkou_a = df['Senkou_A'].iloc[current_idx]
            senkou_b = df['Senkou_B'].iloc[current_idx]

            # Cloud boundaries
            cloud_top = max(senkou_a, senkou_b) if pd.notna(senkou_a) and pd.notna(senkou_b) else None
            cloud_bottom = min(senkou_a, senkou_b) if pd.notna(senkou_a) and pd.notna(senkou_b) else None

            # Signals
            signals = []
            strength = 0

            # 1. Price vs Cloud
            if cloud_top and cloud_bottom:
                if price > cloud_top:
                    signals.append("Price above cloud")
                    strength += 0.3
                elif price < cloud_bottom:
                    signals.append("Price below cloud")
                    strength -= 0.3
                else:
                    signals.append("Price in cloud")

            # 2. Cloud color (bullish when Senkou A > Senkou B)
            if pd.notna(senkou_a) and pd.notna(senkou_b):
                if senkou_a > senkou_b:
                    signals.append("Bullish cloud")
                    strength += 0.2
                else:
                    signals.append("Bearish cloud")
                    strength -= 0.2

            # 3. TK Cross
            if pd.notna(tenkan) and pd.notna(kijun):
                if tenkan > kijun:
                    prev_tenkan = df['Tenkan'].iloc[-2]
                    prev_kijun = df['Kijun'].iloc[-2]
                    if prev_tenkan <= prev_kijun:
                        signals.append("TK bullish cross")
                        strength += 0.3
                    else:
                        signals.append("Tenkan > Kijun")
                        strength += 0.1
                else:
                    prev_tenkan = df['Tenkan'].iloc[-2]
                    prev_kijun = df['Kijun'].iloc[-2]
                    if prev_tenkan >= prev_kijun:
                        signals.append("TK bearish cross")
                        strength -= 0.3
                    else:
                        signals.append("Tenkan < Kijun")
                        strength -= 0.1

            # 4. Price vs Kijun (support/resistance)
            if pd.notna(kijun):
                if price > kijun * 1.01:
                    signals.append("Price above Kijun")
                    strength += 0.1
                elif price < kijun * 0.99:
                    signals.append("Price below Kijun")
                    strength -= 0.1

            # Determine action
            if strength >= 0.4:
                action = "BUY"
            elif strength <= -0.4:
                action = "SELL"
            else:
                action = "NEUTRAL"

            confidence = min(abs(strength), 1.0)

            return self._create_signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                reason=f"Ichimoku: {', '.join(signals)}",
                data={
                    "price": float(price),
                    "tenkan": float(tenkan) if pd.notna(tenkan) else None,
                    "kijun": float(kijun) if pd.notna(kijun) else None,
                    "senkou_a": float(senkou_a) if pd.notna(senkou_a) else None,
                    "senkou_b": float(senkou_b) if pd.notna(senkou_b) else None,
                    "cloud_top": float(cloud_top) if cloud_top else None,
                    "cloud_bottom": float(cloud_bottom) if cloud_bottom else None,
                    "signals": signals
                }
            )

        except Exception as e:
            logger.error(f"Error generating Ichimoku signal for {symbol}: {str(e)}")
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
