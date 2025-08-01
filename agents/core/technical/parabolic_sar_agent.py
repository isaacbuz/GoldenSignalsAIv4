"""
Parabolic SAR Agent
Stop and Reverse indicator for trend following
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class ParabolicSARAgent:
    """
    Parabolic SAR trading agent

    Signals:
    - SAR flip (trend reversal)
    - Price distance from SAR (trend strength)
    - Acceleration factor changes
    """

    def __init__(self):
        self.name = "ParabolicSARAgent"
        self.initial_af = 0.02  # Initial acceleration factor
        self.max_af = 0.2      # Maximum acceleration factor
        self.af_increment = 0.02  # AF increment

    def calculate_psar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Parabolic SAR"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        # Initialize
        psar = np.zeros(len(df))
        trend = np.zeros(len(df))  # 1 for uptrend, -1 for downtrend
        af = np.zeros(len(df))
        ep = np.zeros(len(df))  # Extreme Point

        # Start with uptrend assumption
        psar[0] = low[0]
        trend[0] = 1
        af[0] = self.initial_af
        ep[0] = high[0]

        for i in range(1, len(df)):
            # Previous values
            prev_psar = psar[i-1]
            prev_trend = trend[i-1]
            prev_af = af[i-1]
            prev_ep = ep[i-1]

            # Calculate new PSAR
            if prev_trend == 1:  # Uptrend
                psar[i] = prev_psar + prev_af * (prev_ep - prev_psar)

                # Make sure PSAR is below the last two lows
                psar[i] = min(psar[i], low[i-1])
                if i > 1:
                    psar[i] = min(psar[i], low[i-2])

                # Check for trend reversal
                if low[i] <= psar[i]:
                    trend[i] = -1
                    psar[i] = prev_ep
                    ep[i] = low[i]
                    af[i] = self.initial_af
                else:
                    trend[i] = 1
                    ep[i] = max(prev_ep, high[i])

                    # Update AF if new high
                    if high[i] > prev_ep:
                        af[i] = min(prev_af + self.af_increment, self.max_af)
                    else:
                        af[i] = prev_af

            else:  # Downtrend
                psar[i] = prev_psar - prev_af * (prev_psar - prev_ep)

                # Make sure PSAR is above the last two highs
                psar[i] = max(psar[i], high[i-1])
                if i > 1:
                    psar[i] = max(psar[i], high[i-2])

                # Check for trend reversal
                if high[i] >= psar[i]:
                    trend[i] = 1
                    psar[i] = prev_ep
                    ep[i] = high[i]
                    af[i] = self.initial_af
                else:
                    trend[i] = -1
                    ep[i] = min(prev_ep, low[i])

                    # Update AF if new low
                    if low[i] < prev_ep:
                        af[i] = min(prev_af + self.af_increment, self.max_af)
                    else:
                        af[i] = prev_af

        # Add to dataframe
        df['PSAR'] = psar
        df['PSAR_Trend'] = trend
        df['PSAR_AF'] = af
        df['PSAR_EP'] = ep

        return df

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate Parabolic SAR trading signal"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2mo", interval="1d")

            if df.empty or len(df) < 20:
                return self._create_signal(symbol, "NEUTRAL", 0, "Insufficient data")

            # Calculate Parabolic SAR
            df = self.calculate_psar(df)

            # Get current values
            current_idx = -1
            current_price = df['Close'].iloc[current_idx]
            current_psar = df['PSAR'].iloc[current_idx]
            current_trend = df['PSAR_Trend'].iloc[current_idx]
            current_af = df['PSAR_AF'].iloc[current_idx]

            # Previous values for flip detection
            prev_trend = df['PSAR_Trend'].iloc[-2]

            signals = []
            strength = 0

            # 1. Trend Direction
            if current_trend == 1:
                signals.append("Uptrend (PSAR below price)")
                base_strength = 0.3
            else:
                signals.append("Downtrend (PSAR above price)")
                base_strength = -0.3

            # 2. Trend Reversal Detection
            if current_trend != prev_trend:
                if current_trend == 1:
                    signals.append("Bullish SAR flip")
                    strength = 0.6  # Strong signal on reversal
                else:
                    signals.append("Bearish SAR flip")
                    strength = -0.6  # Strong signal on reversal
            else:
                strength = base_strength

                # 3. Distance from SAR (trend strength)
                distance_pct = abs(current_price - current_psar) / current_price * 100

                if distance_pct > 3:
                    signals.append(f"Strong trend ({distance_pct:.1f}% from SAR)")
                    strength *= 1.2
                elif distance_pct < 1:
                    signals.append(f"Weak trend ({distance_pct:.1f}% from SAR)")
                    strength *= 0.8
                else:
                    signals.append(f"Moderate trend ({distance_pct:.1f}% from SAR)")

            # 4. Acceleration Factor Analysis
            if current_af >= 0.1:
                signals.append(f"High acceleration (AF={current_af:.3f})")
                strength *= 1.1
            elif current_af <= 0.04:
                signals.append(f"Low acceleration (AF={current_af:.3f})")
                strength *= 0.9

            # 5. Trend Duration
            trend_duration = 0
            for i in range(2, min(20, len(df))):
                if df['PSAR_Trend'].iloc[-i] == current_trend:
                    trend_duration += 1
                else:
                    break

            if trend_duration > 10:
                signals.append(f"Mature trend ({trend_duration} days)")
                strength *= 0.9  # Reduce strength for old trends
            elif trend_duration < 3:
                signals.append(f"New trend ({trend_duration} days)")
                strength *= 1.1  # Increase strength for new trends

            # Volume confirmation
            avg_volume = df['Volume'].tail(20).mean()
            current_volume = df['Volume'].iloc[-1]
            if current_volume > avg_volume * 1.5:
                signals.append("High volume confirmation")
                strength *= 1.1

            # Determine action
            if strength >= 0.35:
                action = "BUY"
            elif strength <= -0.35:
                action = "SELL"
            else:
                action = "NEUTRAL"

            confidence = min(abs(strength), 1.0)

            return self._create_signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                reason=f"PSAR: {', '.join(signals)}",
                data={
                    "price": float(current_price),
                    "psar": float(current_psar),
                    "trend": "Up" if current_trend == 1 else "Down",
                    "acceleration_factor": float(current_af),
                    "distance_from_sar": float(abs(current_price - current_psar)),
                    "distance_pct": float(abs(current_price - current_psar) / current_price * 100),
                    "trend_duration": trend_duration,
                    "signals": signals
                }
            )

        except Exception as e:
            logger.error(f"Error generating Parabolic SAR signal for {symbol}: {str(e)}")
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
