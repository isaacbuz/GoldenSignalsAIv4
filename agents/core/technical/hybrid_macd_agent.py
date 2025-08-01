"""
Hybrid MACD Agent - Independent + Collaborative MACD Analysis
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.common.data_bus import SharedDataTypes
from agents.common.hybrid_agent_base import HybridAgent

logger = logging.getLogger(__name__)

class HybridMACDAgent(HybridAgent):
    """
    MACD Agent with dual signal generation

    Independent: Pure MACD crossovers and divergences
    Collaborative: MACD + volume, patterns, support/resistance context
    """

    def __init__(self, data_bus=None, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("HybridMACDAgent", data_bus)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def analyze_independent(self, symbol: str, data: Any = None) -> Dict[str, Any]:
        """Pure MACD analysis without external context"""
        try:
            # Fetch data if not provided
            if data is None:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="6mo")

            if len(data) < self.slow_period + self.signal_period:
                return self._create_signal("HOLD", 0.0, "Insufficient data for MACD")

            # Calculate MACD
            macd_data = self._calculate_macd(data)

            # Get current values
            current_macd = macd_data['macd'].iloc[-1]
            current_signal = macd_data['signal'].iloc[-1]
            current_histogram = macd_data['histogram'].iloc[-1]

            prev_macd = macd_data['macd'].iloc[-2]
            prev_signal = macd_data['signal'].iloc[-2]
            prev_histogram = macd_data['histogram'].iloc[-2]

            # Analyze MACD
            action = "HOLD"
            confidence = 0.0
            reasoning = []

            # Crossover signals
            if prev_macd < prev_signal and current_macd > current_signal:
                # Bullish crossover
                action = "BUY"
                confidence = 0.7
                reasoning.append("MACD bullish crossover")

                # Check if crossover below zero (stronger signal)
                if current_macd < 0:
                    confidence += 0.1
                    reasoning.append("Crossover below zero line")

            elif prev_macd > prev_signal and current_macd < current_signal:
                # Bearish crossover
                action = "SELL"
                confidence = 0.7
                reasoning.append("MACD bearish crossover")

                # Check if crossover above zero (stronger signal)
                if current_macd > 0:
                    confidence += 0.1
                    reasoning.append("Crossover above zero line")

            # Histogram momentum
            histogram_momentum = self._analyze_histogram_momentum(macd_data['histogram'])
            if histogram_momentum['trend'] == 'strengthening':
                if current_histogram > 0:
                    if action == "HOLD":
                        action = "BUY"
                        confidence = 0.5
                    elif action == "BUY":
                        confidence += 0.05
                    reasoning.append("MACD momentum strengthening (bullish)")
                else:
                    if action == "HOLD":
                        action = "SELL"
                        confidence = 0.5
                    elif action == "SELL":
                        confidence += 0.05
                    reasoning.append("MACD momentum strengthening (bearish)")

            elif histogram_momentum['trend'] == 'weakening':
                confidence *= 0.9
                reasoning.append("MACD momentum weakening")

            # Check for divergences
            divergence = self._check_macd_divergence(data, macd_data)
            if divergence['type'] == 'bullish':
                if action == "SELL":
                    action = "HOLD"
                    confidence = 0.3
                elif action == "HOLD":
                    action = "BUY"
                    confidence = 0.6
                else:
                    confidence += 0.1
                reasoning.append("Bullish MACD divergence")

            elif divergence['type'] == 'bearish':
                if action == "BUY":
                    action = "HOLD"
                    confidence = 0.3
                elif action == "HOLD":
                    action = "SELL"
                    confidence = 0.6
                else:
                    confidence += 0.1
                reasoning.append("Bearish MACD divergence")

            # Share momentum state
            if self.data_bus:
                self._share_momentum_state(symbol, current_macd, current_histogram, histogram_momentum)

            confidence = min(confidence, 0.85)

            return self._create_signal(
                action,
                confidence,
                " | ".join(reasoning) if reasoning else "MACD neutral",
                {
                    'macd': float(current_macd),
                    'signal': float(current_signal),
                    'histogram': float(current_histogram),
                    'histogram_momentum': histogram_momentum,
                    'divergence': divergence
                }
            )

        except Exception as e:
            logger.error(f"Error in MACD independent analysis: {e}")
            return self._create_signal("HOLD", 0.0, f"Error: {str(e)}")

    def analyze_collaborative(self, symbol: str, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """MACD analysis enhanced with market context"""
        try:
            # Start with independent analysis
            base_signal = self.analyze_independent(symbol, data)

            # Get context
            volume_data = context['data'].get(SharedDataTypes.VOLUME_SPIKES, {})
            patterns = context['data'].get(SharedDataTypes.PRICE_PATTERNS, {})
            support_resistance = context['data'].get(SharedDataTypes.SUPPORT_RESISTANCE, {})
            trend_direction = context['data'].get(SharedDataTypes.TREND_DIRECTION, {})

            # Enhanced analysis
            action = base_signal['action']
            confidence = base_signal['confidence']
            reasons = [base_signal['metadata']['reasoning']]
            adjustments = []

            macd_indicators = base_signal['metadata']['indicators']

            # Volume confirmation
            if volume_data:
                for agent, vol_info in volume_data.items():
                    spike_type = vol_info['data'].get('spike_type')

                    if action == "BUY" and spike_type == 'bullish_spike':
                        confidence += 0.15
                        adjustments.append("Volume confirms MACD signal")
                    elif action == "SELL" and spike_type == 'bearish_spike':
                        confidence += 0.15
                        adjustments.append("Volume confirms MACD signal")
                    elif (action == "BUY" and spike_type == 'bearish_spike') or \
                         (action == "SELL" and spike_type == 'bullish_spike'):
                        confidence *= 0.8
                        adjustments.append("Volume contradicts MACD - caution")

            # Pattern context
            if patterns:
                for agent, pattern_data in patterns.items():
                    if pattern_data['data'].get('bullish_pattern') and action == "BUY":
                        confidence += 0.1
                        adjustments.append("Bullish pattern aligns with MACD")
                    elif pattern_data['data'].get('bearish_pattern') and action == "SELL":
                        confidence += 0.1
                        adjustments.append("Bearish pattern aligns with MACD")

            # Support/Resistance context
            if support_resistance and data is not None:
                current_price = data['Close'].iloc[-1]

                for agent, sr_data in support_resistance.items():
                    levels = sr_data['data']

                    # MACD signal near key levels
                    if levels.get('support_levels') and action == "BUY":
                        near_support = any(abs(current_price - s) / current_price < 0.02
                                         for s in levels['support_levels'])
                        if near_support:
                            confidence += 0.1
                            adjustments.append("MACD buy signal near support")

                    if levels.get('resistance_levels') and action == "SELL":
                        near_resistance = any(abs(current_price - r) / current_price < 0.02
                                            for r in levels['resistance_levels'])
                        if near_resistance:
                            confidence += 0.1
                            adjustments.append("MACD sell signal near resistance")

            # Trend alignment
            if trend_direction:
                for agent, trend_data in trend_direction.items():
                    trend = trend_data['data'].get('direction')

                    if trend == 'up' and action == "BUY":
                        confidence += 0.1
                        adjustments.append("MACD aligns with uptrend")
                    elif trend == 'down' and action == "SELL":
                        confidence += 0.1
                        adjustments.append("MACD aligns with downtrend")
                    elif (trend == 'up' and action == "SELL") or \
                         (trend == 'down' and action == "BUY"):
                        confidence *= 0.7
                        adjustments.append("MACD against trend - possible reversal")

            # Special cases
            if macd_indicators['divergence']['type'] != 'none':
                if len(adjustments) > 0:
                    confidence += 0.05
                    adjustments.append("Divergence with multiple confirmations")
                else:
                    confidence *= 0.9
                    adjustments.append("Divergence lacks confirmation")

            # Cap confidence
            confidence = min(confidence, 0.95)

            # Build enhanced reasoning
            if adjustments:
                reasons.extend(adjustments)

            return self._create_signal(
                action,
                confidence,
                " | ".join(reasons),
                {
                    **macd_indicators,
                    'context_confirmations': len(adjustments),
                    'collaborative_adjustments': adjustments
                }
            )

        except Exception as e:
            logger.error(f"Error in MACD collaborative analysis: {e}")
            return base_signal

    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        # Calculate EMAs
        ema_fast = data['Close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=self.slow_period, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })

    def _analyze_histogram_momentum(self, histogram: pd.Series) -> Dict[str, Any]:
        """Analyze MACD histogram momentum"""
        recent_hist = histogram.tail(5)

        # Check if histogram is expanding or contracting
        if len(recent_hist) < 5:
            return {'trend': 'neutral', 'strength': 0}

        # Calculate trend
        if all(abs(recent_hist.iloc[i]) > abs(recent_hist.iloc[i-1]) for i in range(1, len(recent_hist))):
            trend = 'strengthening'
            strength = abs(recent_hist.iloc[-1] - recent_hist.iloc[0]) / abs(recent_hist.iloc[0]) if recent_hist.iloc[0] != 0 else 1
        elif all(abs(recent_hist.iloc[i]) < abs(recent_hist.iloc[i-1]) for i in range(1, len(recent_hist))):
            trend = 'weakening'
            strength = abs(recent_hist.iloc[-1] - recent_hist.iloc[0]) / abs(recent_hist.iloc[0]) if recent_hist.iloc[0] != 0 else 1
        else:
            trend = 'neutral'
            strength = 0

        return {
            'trend': trend,
            'strength': float(strength),
            'direction': 'bullish' if recent_hist.iloc[-1] > 0 else 'bearish'
        }

    def _check_macd_divergence(self, price_data: pd.DataFrame, macd_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for MACD divergences"""
        if len(price_data) < 30 or len(macd_data) < 30:
            return {'type': 'none', 'strength': 0}

        # Look at recent 30 periods
        recent_prices = price_data['Close'].tail(30)
        recent_macd = macd_data['macd'].tail(30)

        # Find price highs and lows
        price_highs_idx = []
        price_lows_idx = []

        for i in range(2, len(recent_prices)-2):
            if recent_prices.iloc[i] > recent_prices.iloc[i-1] and recent_prices.iloc[i] > recent_prices.iloc[i+1]:
                price_highs_idx.append(i)
            if recent_prices.iloc[i] < recent_prices.iloc[i-1] and recent_prices.iloc[i] < recent_prices.iloc[i+1]:
                price_lows_idx.append(i)

        # Check for divergences
        if len(price_highs_idx) >= 2:
            # Bearish divergence: price makes higher high, MACD makes lower high
            if recent_prices.iloc[price_highs_idx[-1]] > recent_prices.iloc[price_highs_idx[-2]] and \
               recent_macd.iloc[price_highs_idx[-1]] < recent_macd.iloc[price_highs_idx[-2]]:
                return {'type': 'bearish', 'strength': 0.7}

        if len(price_lows_idx) >= 2:
            # Bullish divergence: price makes lower low, MACD makes higher low
            if recent_prices.iloc[price_lows_idx[-1]] < recent_prices.iloc[price_lows_idx[-2]] and \
               recent_macd.iloc[price_lows_idx[-1]] > recent_macd.iloc[price_lows_idx[-2]]:
                return {'type': 'bullish', 'strength': 0.7}

        return {'type': 'none', 'strength': 0}

    def _share_momentum_state(self, symbol: str, macd_value: float, histogram: float, momentum: Dict):
        """Share momentum state via data bus"""
        if not self.data_bus:
            return

        # Determine overall momentum state
        if macd_value > 0 and histogram > 0 and momentum['trend'] == 'strengthening':
            state = 'bullish'
        elif macd_value < 0 and histogram < 0 and momentum['trend'] == 'strengthening':
            state = 'bearish'
        else:
            state = 'neutral'

        self.data_bus.publish(
            self.name,
            symbol,
            SharedDataTypes.MOMENTUM_STATE,
            {
                'state': state,
                'macd_value': macd_value,
                'histogram': histogram,
                'momentum_trend': momentum['trend'],
                'momentum_strength': momentum['strength']
            }
        )

    def _create_signal(self, action: str, confidence: float, reasoning: str,
                      indicators: Dict = None) -> Dict[str, Any]:
        """Create standardized signal"""
        return {
            'action': action,
            'confidence': confidence,
            'metadata': {
                'agent': self.name,
                'reasoning': reasoning,
                'indicators': indicators or {}
            }
        }

    def _get_relevant_context(self, symbol: str) -> Dict[str, Any]:
        """Specify what context MACD agent needs"""
        if self.data_bus:
            return self.data_bus.get_context(symbol, [
                SharedDataTypes.VOLUME_SPIKES,
                SharedDataTypes.PRICE_PATTERNS,
                SharedDataTypes.SUPPORT_RESISTANCE,
                SharedDataTypes.TREND_DIRECTION
            ])
        return {}
