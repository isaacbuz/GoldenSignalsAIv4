"""
Volume Spike Agent - Detects unusual volume spikes that often precede significant price moves.
Analyzes volume patterns, volume-price relationships, and institutional flow.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class VolumeSpikeAgent(BaseAgent):
    """Agent that detects unusual volume spikes and analyzes their implications."""

    def __init__(
        self,
        name: str = "VolumeSpike",
        lookback_period: int = 20,
        spike_threshold: float = 2.0,
        extreme_spike_threshold: float = 5.0,
        volume_ma_period: int = 10,
        price_volume_correlation_period: int = 15,
        min_volume_threshold: int = 1000
    ):
        """
        Initialize Volume Spike agent.

        Args:
            name: Agent name
            lookback_period: Period for volume average calculation
            spike_threshold: Volume spike threshold (2x average)
            extreme_spike_threshold: Extreme spike threshold (5x average)
            volume_ma_period: Moving average period for volume
            price_volume_correlation_period: Period for price-volume correlation
            min_volume_threshold: Minimum volume to consider
        """
        super().__init__(name=name, agent_type="volume")
        self.lookback_period = lookback_period
        self.spike_threshold = spike_threshold
        self.extreme_spike_threshold = extreme_spike_threshold
        self.volume_ma_period = volume_ma_period
        self.price_volume_correlation_period = price_volume_correlation_period
        self.min_volume_threshold = min_volume_threshold

    def calculate_volume_statistics(self, volume: pd.Series) -> Optional[Dict[str, float]]:
        """Calculate volume statistics for spike detection."""
        try:
            if len(volume) < self.lookback_period:
                return None

            recent_volume = volume.tail(self.lookback_period)
            current_volume = volume.iloc[-1]

            # Basic statistics
            avg_volume = recent_volume.mean()
            median_volume = recent_volume.median()
            std_volume = recent_volume.std()

            # Volume spike ratio
            spike_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            # Z-score for volume
            volume_zscore = (current_volume - avg_volume) / std_volume if std_volume > 0 else 0

            # Percentile rank of current volume
            volume_percentile = stats.percentileofscore(recent_volume, current_volume)

            # Volume trend (increasing/decreasing)
            if len(volume) >= 5:
                recent_5_volume = volume.tail(5)
                volume_slope, _, volume_r_value, _, _ = stats.linregress(
                    range(len(recent_5_volume)), recent_5_volume
                )
            else:
                volume_slope = 0
                volume_r_value = 0

            return {
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'median_volume': median_volume,
                'spike_ratio': spike_ratio,
                'volume_zscore': volume_zscore,
                'volume_percentile': volume_percentile,
                'volume_slope': volume_slope,
                'volume_trend_strength': abs(volume_r_value)
            }

        except Exception as e:
            logger.error(f"Volume statistics calculation failed: {str(e)}")
            return None

    def analyze_price_volume_relationship(self, prices: pd.Series, volume: pd.Series) -> Optional[Dict[str, float]]:
        """Analyze the relationship between price and volume changes."""
        try:
            if len(prices) < self.price_volume_correlation_period or len(volume) < self.price_volume_correlation_period:
                return None

            # Calculate price and volume changes
            price_changes = prices.pct_change().tail(self.price_volume_correlation_period)
            volume_changes = volume.pct_change().tail(self.price_volume_correlation_period)

            # Remove NaN values
            valid_data = pd.DataFrame({'price_change': price_changes, 'volume_change': volume_changes}).dropna()

            if len(valid_data) < 5:
                return None

            # Calculate correlation
            correlation = valid_data['price_change'].corr(valid_data['volume_change'])

            # Calculate current price and volume change
            current_price_change = prices.pct_change().iloc[-1]
            current_volume_change = volume.pct_change().iloc[-1]

            # Analyze patterns
            # Positive correlation: volume follows price (normal)
            # Negative correlation: divergence (potentially significant)

            # Check for volume without price movement (accumulation/distribution)
            recent_price_volatility = price_changes.std()
            recent_volume_spike = volume.iloc[-1] / volume.tail(self.lookback_period).mean()

            volume_without_price_move = (
                recent_volume_spike > self.spike_threshold and
                abs(current_price_change) < recent_price_volatility
            )

            return {
                'price_volume_correlation': correlation,
                'current_price_change': current_price_change,
                'current_volume_change': current_volume_change,
                'volume_without_price_move': volume_without_price_move,
                'price_volatility': recent_price_volatility
            }

        except Exception as e:
            logger.error(f"Price-volume relationship analysis failed: {str(e)}")
            return None

    def detect_volume_patterns(self, volume: pd.Series, prices: pd.Series) -> Optional[Dict[str, Any]]:
        """Detect specific volume patterns and their implications."""
        try:
            if len(volume) < self.lookback_period or len(prices) < self.lookback_period:
                return None

            patterns = []
            current_volume = volume.iloc[-1]
            avg_volume = volume.tail(self.lookback_period).mean()

            # Pattern 1: Breakout volume spike
            if len(prices) >= 3:
                price_range = prices.tail(self.lookback_period)
                recent_high = price_range.max()
                recent_low = price_range.min()
                current_price = prices.iloc[-1]

                near_resistance = current_price > recent_high * 0.98
                near_support = current_price < recent_low * 1.02

                if current_volume > avg_volume * self.spike_threshold:
                    if near_resistance:
                        patterns.append({
                            'type': 'breakout_volume',
                            'direction': 'bullish',
                            'strength': current_volume / avg_volume
                        })
                    elif near_support:
                        patterns.append({
                            'type': 'breakdown_volume',
                            'direction': 'bearish',
                            'strength': current_volume / avg_volume
                        })

            # Pattern 2: Climax volume (exhaustion)
            if current_volume > avg_volume * self.extreme_spike_threshold:
                recent_trend = prices.iloc[-1] / prices.iloc[-5] - 1 if len(prices) >= 5 else 0

                if abs(recent_trend) > 0.05:  # 5% move
                    patterns.append({
                        'type': 'climax_volume',
                        'direction': 'reversal_warning',
                        'strength': current_volume / avg_volume,
                        'trend_direction': 'up' if recent_trend > 0 else 'down'
                    })

            # Pattern 3: Accumulation/Distribution
            volume_ma = volume.rolling(window=self.volume_ma_period).mean()

            if len(volume_ma) >= self.volume_ma_period:
                recent_volume_trend = volume_ma.iloc[-1] / volume_ma.iloc[-self.volume_ma_period]
                price_trend = prices.iloc[-1] / prices.iloc[-self.volume_ma_period] - 1

                if recent_volume_trend > 1.2 and abs(price_trend) < 0.03:  # Volume up, price flat
                    if prices.iloc[-1] > prices.tail(self.volume_ma_period).median():
                        patterns.append({
                            'type': 'accumulation',
                            'direction': 'bullish',
                            'strength': recent_volume_trend
                        })
                    else:
                        patterns.append({
                            'type': 'distribution',
                            'direction': 'bearish',
                            'strength': recent_volume_trend
                        })

            # Pattern 4: Volume dry-up (low volume before move)
            if current_volume < avg_volume * 0.5:  # Very low volume
                patterns.append({
                    'type': 'volume_dryup',
                    'direction': 'pre_move_warning',
                    'strength': avg_volume / current_volume
                })

            return {
                'patterns': patterns,
                'pattern_count': len(patterns)
            } if patterns else None

        except Exception as e:
            logger.error(f"Volume pattern detection failed: {str(e)}")
            return None

    def calculate_institutional_flow_indicator(self, volume: pd.Series, prices: pd.Series, highs: pd.Series, lows: pd.Series) -> Optional[float]:
        """Calculate institutional flow indicator based on volume and price action."""
        try:
            if len(volume) < 5 or len(prices) < 5:
                return None

            # Calculate money flow multiplier
            typical_prices = (highs + lows + prices) / 3
            raw_money_flow = typical_prices * volume

            # Positive and negative money flow
            positive_flow = 0
            negative_flow = 0

            for i in range(1, min(len(typical_prices), self.lookback_period)):
                if typical_prices.iloc[i] > typical_prices.iloc[i-1]:
                    positive_flow += raw_money_flow.iloc[i]
                elif typical_prices.iloc[i] < typical_prices.iloc[i-1]:
                    negative_flow += raw_money_flow.iloc[i]

            # Money Flow Index
            if positive_flow + negative_flow > 0:
                money_ratio = positive_flow / negative_flow if negative_flow > 0 else float('inf')
                mfi = 100 - (100 / (1 + money_ratio))
            else:
                mfi = 50

            return float(mfi)

        except Exception as e:
            logger.error(f"Institutional flow calculation failed: {str(e)}")
            return None

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and detect volume spike signals."""
        try:
            if "volume" not in data or "close_prices" not in data:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Volume or price data not found"}
                }

            volume = pd.Series(data["volume"])
            prices = pd.Series(data["close_prices"])
            highs = pd.Series(data.get("high_prices", prices))
            lows = pd.Series(data.get("low_prices", prices))

            if len(volume) < self.lookback_period or volume.iloc[-1] < self.min_volume_threshold:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient volume data or volume too low"}
                }

            # Calculate volume statistics
            volume_stats = self.calculate_volume_statistics(volume)
            if volume_stats is None:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Failed to calculate volume statistics"}
                }

            # Analyze price-volume relationship
            pv_relationship = self.analyze_price_volume_relationship(prices, volume)

            # Detect volume patterns
            patterns = self.detect_volume_patterns(volume, prices)

            # Calculate institutional flow
            institutional_flow = self.calculate_institutional_flow_indicator(volume, prices, highs, lows)

            # Generate signal based on volume spike and patterns
            action = "hold"
            confidence = 0.0
            signal_reasoning = []

            # Volume spike detected
            if volume_stats['spike_ratio'] >= self.spike_threshold:
                signal_reasoning.append(f"Volume spike: {volume_stats['spike_ratio']:.1f}x average")

                # Determine direction based on patterns and price action
                if patterns and patterns['patterns']:
                    for pattern in patterns['patterns']:
                        if pattern['direction'] == 'bullish':
                            action = "buy"
                            confidence += 0.3
                        elif pattern['direction'] == 'bearish':
                            action = "sell"
                            confidence += 0.3
                        elif pattern['direction'] == 'reversal_warning':
                            # Extreme volume suggests possible reversal
                            current_trend = prices.iloc[-1] / prices.iloc[-5] - 1 if len(prices) >= 5 else 0
                            if current_trend > 0:
                                action = "sell"  # Sell after strong up move with extreme volume
                            else:
                                action = "buy"   # Buy after strong down move with extreme volume
                            confidence += 0.4

                # Volume without price movement (accumulation/distribution)
                if pv_relationship and pv_relationship['volume_without_price_move']:
                    if institutional_flow and institutional_flow > 60:
                        action = "buy"
                        signal_reasoning.append("Possible accumulation detected")
                        confidence += 0.2
                    elif institutional_flow and institutional_flow < 40:
                        action = "sell"
                        signal_reasoning.append("Possible distribution detected")
                        confidence += 0.2

                # Boost confidence for extreme spikes
                if volume_stats['spike_ratio'] >= self.extreme_spike_threshold:
                    confidence *= 1.5
                    signal_reasoning.append("Extreme volume spike detected")

            # Volume percentile boost
            if volume_stats['volume_percentile'] > 90:
                confidence *= 1.2
                signal_reasoning.append(f"Volume in {volume_stats['volume_percentile']:.0f}th percentile")

            return {
                "action": action,
                "confidence": min(1.0, confidence),
                "metadata": {
                    "volume_stats": volume_stats,
                    "price_volume_relationship": pv_relationship,
                    "patterns": patterns,
                    "institutional_flow": institutional_flow,
                    "signal_reasoning": signal_reasoning
                }
            }

        except Exception as e:
            logger.error(f"Volume spike signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
