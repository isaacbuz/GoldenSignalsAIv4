"""
Volume Analysis Agent
Analyzes volume patterns to detect unusual activity and potential price movements
"""

import numpy as np
from typing import Dict, Any, List
from agents.unified_base_agent import UnifiedBaseAgent, SignalStrength
import logging

logger = logging.getLogger(__name__)


class VolumeAgent(UnifiedBaseAgent):
    """
    Agent that analyzes volume patterns including:
    - Volume spikes and anomalies
    - Volume-price divergence
    - Accumulation/Distribution patterns
    - On-Balance Volume (OBV) trends
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="VolumeAgent",
            weight=1.0,  # Standard weight
            config=config
        )

        # Configuration
        self.spike_threshold = self.config.get("spike_threshold", 2.0)  # 2x average
        self.lookback_period = self.config.get("lookback_period", 20)
        self.min_volume = self.config.get("min_volume", 100000)

    def get_required_data_fields(self) -> List[str]:
        """Required fields for volume analysis"""
        return ["symbol", "current_price", "historical_data"]

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze volume patterns and generate trading signals
        """
        try:
            historical_data = market_data.get("historical_data", [])

            if len(historical_data) < self.lookback_period:
                return {
                    "signal": 0,
                    "confidence": 0.1,
                    "reasoning": "Insufficient historical data for volume analysis"
                }

            # Extract volume and price data
            volumes = np.array([candle.get("volume", 0) for candle in historical_data])
            closes = np.array([candle.get("close", 0) for candle in historical_data])
            opens = np.array([candle.get("open", 0) for candle in historical_data])
            highs = np.array([candle.get("high", 0) for candle in historical_data])
            lows = np.array([candle.get("low", 0) for candle in historical_data])

            # Calculate volume metrics
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[-self.lookback_period:])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Detect volume spike
            is_spike = volume_ratio > self.spike_threshold

            # Calculate On-Balance Volume (OBV)
            obv = self._calculate_obv(closes, volumes)
            obv_trend = self._calculate_trend(obv[-self.lookback_period:])

            # Calculate Volume-Price Trend (VPT)
            vpt = self._calculate_vpt(closes, volumes)
            vpt_trend = self._calculate_trend(vpt[-self.lookback_period:])

            # Analyze accumulation/distribution
            ad_line = self._calculate_ad_line(highs, lows, closes, volumes)
            ad_trend = self._calculate_trend(ad_line[-self.lookback_period:])

            # Check for volume-price divergence
            price_trend = self._calculate_trend(closes[-self.lookback_period:])
            volume_trend = self._calculate_trend(volumes[-self.lookback_period:])
            divergence = self._detect_divergence(price_trend, volume_trend)

            # Generate signal based on multiple factors
            signal_score = 0.0
            confidence_factors = []
            reasoning_parts = []

            # Volume spike analysis
            if is_spike:
                price_direction = 1 if closes[-1] > closes[-2] else -1
                signal_score += 0.3 * price_direction * (volume_ratio / self.spike_threshold)
                confidence_factors.append(min(0.9, volume_ratio / 3.0))
                reasoning_parts.append(f"Volume spike detected ({volume_ratio:.1f}x average)")

            # OBV trend
            if abs(obv_trend) > 0.1:
                signal_score += 0.25 * obv_trend
                confidence_factors.append(abs(obv_trend))
                reasoning_parts.append(f"OBV trend {'bullish' if obv_trend > 0 else 'bearish'}")

            # Accumulation/Distribution
            if abs(ad_trend) > 0.1:
                signal_score += 0.25 * ad_trend
                confidence_factors.append(abs(ad_trend))
                if ad_trend > 0:
                    reasoning_parts.append("Accumulation pattern detected")
                else:
                    reasoning_parts.append("Distribution pattern detected")

            # Volume-Price divergence
            if divergence != 0:
                signal_score += 0.2 * divergence
                confidence_factors.append(0.7)
                if divergence > 0:
                    reasoning_parts.append("Bullish volume-price divergence")
                else:
                    reasoning_parts.append("Bearish volume-price divergence")

            # Normalize signal score
            signal_score = max(-1.0, min(1.0, signal_score))

            # Calculate confidence
            confidence = np.mean(confidence_factors) if confidence_factors else 0.3
            confidence = max(0.1, min(0.95, confidence))

            # Build reasoning
            if not reasoning_parts:
                reasoning_parts.append("Normal volume patterns")
            reasoning = "; ".join(reasoning_parts)

            # Additional metrics for data
            analysis_data = {
                "current_volume": int(current_volume),
                "average_volume": int(avg_volume),
                "volume_ratio": round(volume_ratio, 2),
                "is_spike": is_spike,
                "obv_trend": round(obv_trend, 3),
                "ad_trend": round(ad_trend, 3),
                "vpt_trend": round(vpt_trend, 3),
                "price_trend": round(price_trend, 3),
                "volume_trend": round(volume_trend, 3),
                "divergence": divergence
            }

            return {
                "signal": signal_score,
                "confidence": confidence,
                "reasoning": reasoning,
                "volume_ratio": volume_ratio,
                "obv_trend": obv_trend,
                "ad_trend": ad_trend,
                "data": analysis_data
            }

        except Exception as e:
            logger.error(f"Volume analysis error: {e}")
            return {
                "signal": 0,
                "confidence": 0.1,
                "reasoning": f"Volume analysis error: {str(e)}"
            }

    def _calculate_obv(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume"""
        obv = np.zeros(len(closes))
        obv[0] = volumes[0]

        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]

        return obv

    def _calculate_vpt(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate Volume-Price Trend"""
        vpt = np.zeros(len(closes))
        vpt[0] = volumes[0]

        for i in range(1, len(closes)):
            price_change = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] > 0 else 0
            vpt[i] = vpt[i-1] + (volumes[i] * price_change)

        return vpt

    def _calculate_ad_line(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """Calculate Accumulation/Distribution Line"""
        ad = np.zeros(len(closes))

        for i in range(len(closes)):
            if highs[i] != lows[i]:
                mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])
                mfv = mfm * volumes[i]
                ad[i] = ad[i-1] + mfv if i > 0 else mfv
            else:
                ad[i] = ad[i-1] if i > 0 else 0

        return ad

    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend using linear regression slope"""
        if len(data) < 2:
            return 0.0

        x = np.arange(len(data))
        # Normalize data
        data_norm = (data - np.mean(data)) / (np.std(data) + 1e-10)

        # Calculate slope
        slope = np.polyfit(x, data_norm, 1)[0]
        return float(slope)

    def _detect_divergence(self, price_trend: float, volume_trend: float) -> int:
        """
        Detect volume-price divergence
        Returns: 1 for bullish divergence, -1 for bearish, 0 for no divergence
        """
        if abs(price_trend) < 0.1 or abs(volume_trend) < 0.1:
            return 0

        # Bullish divergence: price down, volume up
        if price_trend < -0.1 and volume_trend > 0.1:
            return 1

        # Bearish divergence: price up, volume down
        if price_trend > 0.1 and volume_trend < -0.1:
            return -1

        return 0


# For backward compatibility
from typing import Optional
__all__ = ['VolumeAgent']
