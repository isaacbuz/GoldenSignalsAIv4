"""
Volume Profile Agent
Analyzes volume distribution across price levels to identify key support/resistance
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class VolumeProfileAgent:
    """
    Volume Profile trading agent

    Signals:
    - Point of Control (POC) - highest volume price
    - Value Area High/Low (VAH/VAL)
    - Volume nodes and gaps
    - Price acceptance/rejection at levels
    """

    def __init__(self):
        self.name = "VolumeProfileAgent"
        self.lookback_days = 20  # Profile period
        self.value_area_pct = 0.7  # 70% of volume for value area
        self.num_bins = 30  # Price bins for profile

    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile metrics"""
        # Create price bins
        price_range = df['High'].max() - df['Low'].min()
        bin_size = price_range / self.num_bins

        # Initialize profile
        profile = {}

        # Build volume profile
        for idx, row in df.iterrows():
            # Approximate volume distribution across the day's range
            day_low = row['Low']
            day_high = row['High']
            day_volume = row['Volume']

            # Simple distribution - assume volume is evenly distributed
            # In reality, you'd use intraday data
            num_touches = int((day_high - day_low) / bin_size) + 1
            volume_per_level = day_volume / num_touches

            # Assign volume to price levels
            current_price = day_low
            while current_price <= day_high:
                bin_price = round(current_price / bin_size) * bin_size
                if bin_price not in profile:
                    profile[bin_price] = 0
                profile[bin_price] += volume_per_level
                current_price += bin_size

        # Sort by price
        sorted_profile = sorted(profile.items())
        prices = [p[0] for p in sorted_profile]
        volumes = [p[1] for p in sorted_profile]

        # Find Point of Control (highest volume price)
        poc_idx = np.argmax(volumes)
        poc_price = prices[poc_idx]
        poc_volume = volumes[poc_idx]

        # Calculate Value Area
        total_volume = sum(volumes)
        target_volume = total_volume * self.value_area_pct

        # Start from POC and expand outward
        value_area_volume = poc_volume
        upper_idx = poc_idx
        lower_idx = poc_idx

        while value_area_volume < target_volume:
            # Check which side to expand
            can_go_up = upper_idx < len(prices) - 1
            can_go_down = lower_idx > 0

            if can_go_up and can_go_down:
                # Expand to side with more volume
                up_volume = volumes[upper_idx + 1] if can_go_up else 0
                down_volume = volumes[lower_idx - 1] if can_go_down else 0

                if up_volume > down_volume:
                    upper_idx += 1
                    value_area_volume += up_volume
                else:
                    lower_idx -= 1
                    value_area_volume += down_volume
            elif can_go_up:
                upper_idx += 1
                value_area_volume += volumes[upper_idx]
            elif can_go_down:
                lower_idx -= 1
                value_area_volume += volumes[lower_idx]
            else:
                break

        vah = prices[upper_idx]  # Value Area High
        val = prices[lower_idx]  # Value Area Low

        # Identify high/low volume nodes
        avg_volume = np.mean(volumes)
        high_volume_nodes = [(p, v) for p, v in zip(prices, volumes) if v > avg_volume * 1.5]
        low_volume_nodes = [(p, v) for p, v in zip(prices, volumes) if v < avg_volume * 0.5]

        return {
            'poc': poc_price,
            'poc_volume': poc_volume,
            'vah': vah,
            'val': val,
            'value_area_volume_pct': value_area_volume / total_volume,
            'high_volume_nodes': high_volume_nodes,
            'low_volume_nodes': low_volume_nodes,
            'profile': list(zip(prices, volumes))
        }

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate Volume Profile trading signal"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2mo", interval="1d")

            if df.empty or len(df) < self.lookback_days:
                return self._create_signal(symbol, "NEUTRAL", 0, "Insufficient data")

            # Get recent data for profile
            profile_df = df.tail(self.lookback_days)
            current_price = df['Close'].iloc[-1]

            # Calculate volume profile
            vp_data = self.calculate_volume_profile(profile_df)

            poc = vp_data['poc']
            vah = vp_data['vah']
            val = vp_data['val']

            signals = []
            strength = 0

            # 1. Position relative to value area
            if current_price > vah:
                signals.append(f"Price above value area (VAH: ${vah:.2f})")
                # Check if breaking out or overextended
                distance_pct = (current_price - vah) / vah * 100
                if distance_pct < 1:
                    signals.append("Breaking above value area")
                    strength += 0.3
                elif distance_pct > 3:
                    signals.append("Overextended above value area")
                    strength -= 0.2
            elif current_price < val:
                signals.append(f"Price below value area (VAL: ${val:.2f})")
                distance_pct = (val - current_price) / val * 100
                if distance_pct < 1:
                    signals.append("Breaking below value area")
                    strength -= 0.3
                elif distance_pct > 3:
                    signals.append("Overextended below value area")
                    strength += 0.2
            else:
                signals.append("Price within value area")
                strength += 0.1  # Neutral to slightly bullish

            # 2. Distance from POC
            poc_distance = abs(current_price - poc) / poc * 100
            if poc_distance < 0.5:
                signals.append(f"At Point of Control (${poc:.2f})")
                # POC acts as magnet
                strength *= 0.8  # Reduce strength near POC
            elif current_price > poc:
                signals.append(f"Above POC (${poc:.2f})")
                strength += 0.1
            else:
                signals.append(f"Below POC (${poc:.2f})")
                strength -= 0.1

            # 3. Volume node analysis
            # Find nearest high/low volume nodes
            nearest_hvn = None
            nearest_lvn = None

            for price, _ in vp_data['high_volume_nodes']:
                if nearest_hvn is None or abs(price - current_price) < abs(nearest_hvn - current_price):
                    nearest_hvn = price

            for price, _ in vp_data['low_volume_nodes']:
                if nearest_lvn is None or abs(price - current_price) < abs(nearest_lvn - current_price):
                    nearest_lvn = price

            # High volume nodes act as support/resistance
            if nearest_hvn:
                if abs(current_price - nearest_hvn) / current_price < 0.01:
                    if current_price > nearest_hvn:
                        signals.append(f"Testing HVN support at ${nearest_hvn:.2f}")
                        strength += 0.2
                    else:
                        signals.append(f"Testing HVN resistance at ${nearest_hvn:.2f}")
                        strength -= 0.2

            # Low volume nodes are areas of quick movement
            if nearest_lvn and abs(current_price - nearest_lvn) / current_price < 0.02:
                signals.append(f"In low volume gap near ${nearest_lvn:.2f}")
                # Price tends to move quickly through LVN
                recent_move = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
                if recent_move > 0:
                    strength += 0.2  # Continue upward through gap
                else:
                    strength -= 0.2  # Continue downward through gap

            # 4. Recent volume confirmation
            recent_volume = df['Volume'].tail(5).mean()
            avg_volume = df['Volume'].tail(20).mean()

            if recent_volume > avg_volume * 1.3:
                signals.append("High volume confirmation")
                strength *= 1.2
            elif recent_volume < avg_volume * 0.7:
                signals.append("Low volume warning")
                strength *= 0.8

            # 5. Trend context
            sma_20 = df['Close'].tail(20).mean()
            if current_price > sma_20 and current_price > poc:
                signals.append("Uptrend with volume support")
                strength += 0.1
            elif current_price < sma_20 and current_price < poc:
                signals.append("Downtrend with volume resistance")
                strength -= 0.1

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
                reason=f"Volume Profile: {', '.join(signals)}",
                data={
                    "price": float(current_price),
                    "poc": float(poc),
                    "vah": float(vah),
                    "val": float(val),
                    "value_area_pct": float(vp_data['value_area_volume_pct'] * 100),
                    "nearest_hvn": float(nearest_hvn) if nearest_hvn else None,
                    "nearest_lvn": float(nearest_lvn) if nearest_lvn else None,
                    "signals": signals
                }
            )

        except Exception as e:
            logger.error(f"Error generating Volume Profile signal for {symbol}: {str(e)}")
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
