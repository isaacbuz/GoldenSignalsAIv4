"""
Hybrid Bollinger Bands Agent - Independent + Collaborative Analysis
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

class HybridBollingerAgent(HybridAgent):
    """
    Bollinger Bands Agent with dual signal generation

    Independent: Pure BB squeeze, band touches, and volatility analysis
    Collaborative: BB + volume, patterns, momentum for enhanced signals
    """

    def __init__(self, data_bus=None, period: int = 20, std_dev: float = 2.0):
        super().__init__("HybridBollingerAgent", data_bus)
        self.period = period
        self.std_dev = std_dev

    def analyze_independent(self, symbol: str, data: Any = None) -> Dict[str, Any]:
        """Pure Bollinger Bands analysis without external context"""
        try:
            # Fetch data if not provided
            if data is None:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="3mo")

            if len(data) < self.period:
                return self._create_signal("HOLD", 0.0, "Insufficient data for Bollinger Bands")

            # Calculate Bollinger Bands
            bb_data = self._calculate_bollinger_bands(data)

            # Get current values
            current_price = data['Close'].iloc[-1]
            upper_band = bb_data['upper'].iloc[-1]
            middle_band = bb_data['middle'].iloc[-1]
            lower_band = bb_data['lower'].iloc[-1]
            band_width = bb_data['width'].iloc[-1]

            # Calculate position within bands
            position_in_bands = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5

            # Analyze Bollinger Bands
            action = "HOLD"
            confidence = 0.0
            reasoning = []

            # Band touches and penetrations
            if current_price <= lower_band:
                action = "BUY"
                confidence = 0.7
                reasoning.append("Price at/below lower Bollinger Band")

                # Check if price is recovering
                if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    confidence += 0.1
                    reasoning.append("Price bouncing from lower band")

            elif current_price >= upper_band:
                action = "SELL"
                confidence = 0.7
                reasoning.append("Price at/above upper Bollinger Band")

                # Check if price is reversing
                if data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                    confidence += 0.1
                    reasoning.append("Price reversing from upper band")

            # Squeeze detection
            squeeze_data = self._detect_squeeze(bb_data)
            if squeeze_data['is_squeeze']:
                if action == "HOLD":
                    # During squeeze, wait for breakout
                    confidence = 0.3
                    reasoning.append("Bollinger Band squeeze - waiting for breakout")
                else:
                    # Reduce confidence during squeeze
                    confidence *= 0.7
                    reasoning.append("Signal during BB squeeze - caution")

            elif squeeze_data['post_squeeze']:
                # Just exited squeeze
                if squeeze_data['breakout_direction'] == 'up':
                    if action != "SELL":
                        action = "BUY"
                        confidence = 0.75
                        reasoning.append("Bullish breakout from BB squeeze")
                elif squeeze_data['breakout_direction'] == 'down':
                    if action != "BUY":
                        action = "SELL"
                        confidence = 0.75
                        reasoning.append("Bearish breakout from BB squeeze")

            # Band walk detection
            band_walk = self._detect_band_walk(data, bb_data)
            if band_walk['upper_walk']:
                if action == "BUY":
                    action = "HOLD"
                    confidence = 0.3
                    reasoning.append("Upper band walk - trend may be exhausted")
                elif action == "SELL":
                    confidence *= 0.8
                    reasoning.append("Upper band walk supports reversal")

            elif band_walk['lower_walk']:
                if action == "SELL":
                    action = "HOLD"
                    confidence = 0.3
                    reasoning.append("Lower band walk - trend may be exhausted")
                elif action == "BUY":
                    confidence *= 0.8
                    reasoning.append("Lower band walk supports reversal")

            # Share volatility state
            if self.data_bus:
                self._share_volatility_state(symbol, band_width, squeeze_data, position_in_bands)

            confidence = min(confidence, 0.85)

            return self._create_signal(
                action,
                confidence,
                " | ".join(reasoning) if reasoning else "Bollinger Bands neutral",
                {
                    'upper_band': float(upper_band),
                    'middle_band': float(middle_band),
                    'lower_band': float(lower_band),
                    'band_width': float(band_width),
                    'position_in_bands': float(position_in_bands),
                    'squeeze': squeeze_data,
                    'band_walk': band_walk
                }
            )

        except Exception as e:
            logger.error(f"Error in Bollinger independent analysis: {e}")
            return self._create_signal("HOLD", 0.0, f"Error: {str(e)}")

    def analyze_collaborative(self, symbol: str, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Bollinger Bands analysis enhanced with market context"""
        try:
            # Start with independent analysis
            base_signal = self.analyze_independent(symbol, data)

            # Get context
            volume_data = context['data'].get(SharedDataTypes.VOLUME_SPIKES, {})
            momentum_state = context['data'].get(SharedDataTypes.MOMENTUM_STATE, {})
            patterns = context['data'].get(SharedDataTypes.PRICE_PATTERNS, {})
            market_regime = context['data'].get(SharedDataTypes.MARKET_REGIME, {})

            # Enhanced analysis
            action = base_signal['action']
            confidence = base_signal['confidence']
            reasons = [base_signal['metadata']['reasoning']]
            adjustments = []

            bb_indicators = base_signal['metadata']['indicators']
            position_in_bands = bb_indicators['position_in_bands']

            # Volume confirmation at bands
            if volume_data:
                for agent, vol_info in volume_data.items():
                    spike_type = vol_info['data'].get('spike_type')

                    if position_in_bands < 0.2 and spike_type == 'bullish_spike':
                        confidence += 0.15
                        adjustments.append("Volume spike confirms lower band bounce")
                    elif position_in_bands > 0.8 and spike_type == 'bearish_spike':
                        confidence += 0.15
                        adjustments.append("Volume spike confirms upper band rejection")
                    elif bb_indicators['squeeze']['is_squeeze'] and spike_type in ['bullish_spike', 'bearish_spike']:
                        confidence += 0.1
                        adjustments.append("Volume surge during BB squeeze - breakout likely")

            # Momentum alignment
            if momentum_state:
                for agent, mom_data in momentum_state.items():
                    momentum = mom_data['data'].get('state', '')

                    if momentum == 'bullish' and action == "BUY":
                        confidence += 0.1
                        adjustments.append("Momentum confirms BB signal")
                    elif momentum == 'bearish' and action == "SELL":
                        confidence += 0.1
                        adjustments.append("Momentum confirms BB signal")
                    elif bb_indicators['squeeze']['is_squeeze']:
                        # During squeeze, momentum can indicate breakout direction
                        if momentum == 'bullish':
                            adjustments.append("Bullish momentum during squeeze")
                        elif momentum == 'bearish':
                            adjustments.append("Bearish momentum during squeeze")

            # Pattern context
            if patterns:
                for agent, pattern_data in patterns.items():
                    # Reversal patterns at bands
                    if pattern_data['data'].get('reversal_pattern'):
                        if position_in_bands < 0.2 and action == "BUY":
                            confidence += 0.1
                            adjustments.append("Reversal pattern at lower band")
                        elif position_in_bands > 0.8 and action == "SELL":
                            confidence += 0.1
                            adjustments.append("Reversal pattern at upper band")

            # Market regime adjustments
            if market_regime:
                for agent, regime_data in market_regime.items():
                    regime = regime_data['data'].get('regime', '')

                    if 'trending' in regime:
                        # In trending markets, band touches are less reliable
                        if bb_indicators['band_walk']['upper_walk'] or bb_indicators['band_walk']['lower_walk']:
                            confidence *= 0.8
                            adjustments.append("Band walk in trending market - reduced confidence")
                    elif 'ranging' in regime:
                        # In ranging markets, band touches are more reliable
                        if position_in_bands < 0.2 or position_in_bands > 0.8:
                            confidence += 0.1
                            adjustments.append("Band touch in ranging market - higher reliability")

            # Special squeeze considerations
            if bb_indicators['squeeze']['is_squeeze']:
                if len(adjustments) < 2:
                    confidence *= 0.8
                    adjustments.append("Squeeze without confirmations - wait for breakout")
                else:
                    confidence += 0.05
                    adjustments.append("Squeeze with multiple confirmations")

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
                    **bb_indicators,
                    'context_confirmations': len(adjustments),
                    'collaborative_adjustments': adjustments
                }
            )

        except Exception as e:
            logger.error(f"Error in Bollinger collaborative analysis: {e}")
            return base_signal

    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        # Middle band (SMA)
        middle = data['Close'].rolling(window=self.period).mean()

        # Standard deviation
        std = data['Close'].rolling(window=self.period).std()

        # Upper and lower bands
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)

        # Band width and %B
        width = upper - lower
        percent_b = (data['Close'] - lower) / (upper - lower)

        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'percent_b': percent_b,
            'std': std
        })

    def _detect_squeeze(self, bb_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Bollinger Band squeeze"""
        # Calculate historical band width
        recent_width = bb_data['width'].tail(20)
        if len(recent_width) < 20:
            return {'is_squeeze': False, 'post_squeeze': False, 'squeeze_duration': 0}

        # Current and average width
        current_width = recent_width.iloc[-1]
        avg_width = recent_width.mean()
        min_width = recent_width.min()

        # Detect squeeze (width < 50% of average)
        is_squeeze = current_width < avg_width * 0.5

        # Check if just exited squeeze
        post_squeeze = False
        breakout_direction = None

        if not is_squeeze and recent_width.iloc[-2] < avg_width * 0.5:
            post_squeeze = True
            # Determine breakout direction
            if bb_data['percent_b'].iloc[-1] > 0.5:
                breakout_direction = 'up'
            else:
                breakout_direction = 'down'

        # Calculate squeeze duration
        squeeze_duration = 0
        for i in range(len(recent_width)-1, -1, -1):
            if recent_width.iloc[i] < avg_width * 0.5:
                squeeze_duration += 1
            else:
                break

        return {
            'is_squeeze': is_squeeze,
            'post_squeeze': post_squeeze,
            'squeeze_duration': squeeze_duration,
            'breakout_direction': breakout_direction,
            'width_ratio': float(current_width / avg_width) if avg_width > 0 else 1
        }

    def _detect_band_walk(self, price_data: pd.DataFrame, bb_data: pd.DataFrame) -> Dict[str, bool]:
        """Detect if price is walking the bands"""
        recent_closes = price_data['Close'].tail(5)
        recent_upper = bb_data['upper'].tail(5)
        recent_lower = bb_data['lower'].tail(5)

        # Upper band walk: multiple touches of upper band
        upper_touches = sum(1 for i in range(len(recent_closes))
                          if recent_closes.iloc[i] >= recent_upper.iloc[i] * 0.99)

        # Lower band walk: multiple touches of lower band
        lower_touches = sum(1 for i in range(len(recent_closes))
                          if recent_closes.iloc[i] <= recent_lower.iloc[i] * 1.01)

        return {
            'upper_walk': upper_touches >= 3,
            'lower_walk': lower_touches >= 3,
            'upper_touches': upper_touches,
            'lower_touches': lower_touches
        }

    def _share_volatility_state(self, symbol: str, band_width: float, squeeze_data: Dict, position: float):
        """Share volatility state via data bus"""
        if not self.data_bus:
            return

        # Determine volatility state
        if squeeze_data['is_squeeze']:
            state = 'low_volatility'
        elif band_width > band_width * 1.5:  # This comparison doesn't make sense, fixing
            state = 'high_volatility'
        else:
            state = 'normal_volatility'

        self.data_bus.publish(
            self.name,
            symbol,
            SharedDataTypes.VOLATILITY_STATE,
            {
                'state': state,
                'band_width': band_width,
                'squeeze': squeeze_data['is_squeeze'],
                'position_in_bands': position
            }
        )

        # Also share overbought/oversold based on band position
        if position > 0.9:
            ob_os_state = 'overbought'
        elif position < 0.1:
            ob_os_state = 'oversold'
        else:
            ob_os_state = 'neutral'

        self.data_bus.publish(
            self.name,
            symbol,
            SharedDataTypes.OVERBOUGHT_OVERSOLD,
            {
                'state': ob_os_state,
                'bb_position': position,
                'upper_band_touch': position > 0.95,
                'lower_band_touch': position < 0.05
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
        """Specify what context Bollinger agent needs"""
        if self.data_bus:
            return self.data_bus.get_context(symbol, [
                SharedDataTypes.VOLUME_SPIKES,
                SharedDataTypes.MOMENTUM_STATE,
                SharedDataTypes.PRICE_PATTERNS,
                SharedDataTypes.MARKET_REGIME
            ])
        return {}
