"""
Enhanced Pattern Recognition Agent
Detects chart patterns and generates trading signals using unified base
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from agents.unified_base_agent import SignalStrength, UnifiedBaseAgent
from scipy import stats
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


class PatternAgent(UnifiedBaseAgent):
    """
    Enhanced agent that detects chart patterns including:
    - Double top/bottom
    - Head and shoulders (regular and inverse)
    - Triangles (ascending, descending, symmetric)
    - Flags and pennants
    - Wedges (rising and falling)
    - Channels
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="PatternAgent",
            weight=1.3,  # Higher weight due to pattern reliability
            config=config
        )

        # Configuration
        self.min_pattern_bars = self.config.get("min_pattern_bars", 10)
        self.max_pattern_bars = self.config.get("max_pattern_bars", 50)
        self.tolerance = self.config.get("tolerance", 0.02)  # 2% price tolerance
        self.volume_confirmation = self.config.get("volume_confirmation", True)
        self.peak_order = self.config.get("peak_order", 5)  # Sensitivity for peak detection

    def get_required_data_fields(self) -> List[str]:
        """Required fields for pattern analysis"""
        return ["symbol", "current_price", "historical_data"]

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data for chart patterns
        """
        try:
            historical_data = market_data.get("historical_data", [])

            if len(historical_data) < self.min_pattern_bars:
                return {
                    "signal": 0,
                    "confidence": 0.1,
                    "reasoning": "Insufficient data for pattern recognition"
                }

            # Convert to pandas series for easier analysis
            df = pd.DataFrame(historical_data)
            prices = df['close']
            highs = df['high']
            lows = df['low']
            volumes = df['volume'] if 'volume' in df else None

            # Find peaks and troughs
            peaks, troughs = self._find_peaks_troughs(prices)

            # Detect all patterns
            patterns = []

            # Classic reversal patterns
            double_top = self._detect_double_top(prices, peaks)
            if double_top:
                patterns.append(double_top)

            double_bottom = self._detect_double_bottom(prices, troughs)
            if double_bottom:
                patterns.append(double_bottom)

            head_shoulders = self._detect_head_shoulders(prices, peaks, troughs)
            if head_shoulders:
                patterns.append(head_shoulders)

            # Continuation patterns
            triangle = self._detect_triangle(prices, highs, lows)
            if triangle:
                patterns.append(triangle)

            flag = self._detect_flag(prices, volumes)
            if flag:
                patterns.append(flag)

            wedge = self._detect_wedge(prices, highs, lows)
            if wedge:
                patterns.append(wedge)

            channel = self._detect_channel(prices, highs, lows)
            if channel:
                patterns.append(channel)

            # Select the most confident pattern
            if patterns:
                best_pattern = max(patterns, key=lambda p: p['confidence'])

                # Convert pattern signal to score
                signal_map = {
                    'strong_buy': 0.8,
                    'buy': 0.5,
                    'hold': 0,
                    'sell': -0.5,
                    'strong_sell': -0.8
                }

                signal_score = signal_map.get(best_pattern['signal'], 0)

                # Build comprehensive reasoning
                reasoning_parts = [f"{best_pattern['pattern']} pattern detected"]
                if best_pattern.get('breakout_level'):
                    reasoning_parts.append(f"Breakout level: {best_pattern['breakout_level']:.2f}")
                if best_pattern.get('target'):
                    reasoning_parts.append(f"Target: {best_pattern['target']:.2f}")
                if best_pattern.get('volume_confirmed'):
                    reasoning_parts.append("Volume confirmed")

                reasoning = "; ".join(reasoning_parts)

                return {
                    "signal": signal_score,
                    "confidence": best_pattern['confidence'],
                    "reasoning": reasoning,
                    "pattern": best_pattern['pattern'],
                    "all_patterns": patterns,
                    "pattern_count": len(patterns),
                    "data": best_pattern
                }
            else:
                return {
                    "signal": 0,
                    "confidence": 0.3,
                    "reasoning": "No significant patterns detected",
                    "pattern_count": 0
                }

        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return {
                "signal": 0,
                "confidence": 0.1,
                "reasoning": f"Pattern analysis error: {str(e)}"
            }

    def _find_peaks_troughs(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks and troughs in price series"""
        try:
            peaks = argrelextrema(prices.values, np.greater, order=self.peak_order)[0]
            troughs = argrelextrema(prices.values, np.less, order=self.peak_order)[0]
            return peaks, troughs
        except Exception as e:
            logger.error(f"Peak/trough detection failed: {e}")
            return np.array([]), np.array([])

    def _detect_double_top(self, prices: pd.Series, peaks: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect double top pattern"""
        if len(peaks) < 2:
            return None

        # Look at recent peaks
        recent_peaks = peaks[-5:] if len(peaks) > 5 else peaks

        for i in range(len(recent_peaks) - 1):
            peak1_idx, peak2_idx = recent_peaks[i], recent_peaks[i + 1]
            peak1_price, peak2_price = prices.iloc[peak1_idx], prices.iloc[peak2_idx]

            # Check if peaks are similar
            height_diff = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)

            if height_diff <= self.tolerance:
                # Find valley between peaks
                valley_idx = prices.iloc[peak1_idx:peak2_idx].idxmin()
                valley_price = prices.iloc[valley_idx]

                # Calculate pattern metrics
                pattern_height = (max(peak1_price, peak2_price) - valley_price) / valley_price
                current_price = prices.iloc[-1]

                # Check if price is breaking below neckline
                if current_price < valley_price * 0.98:
                    confidence = min(0.9, 0.5 + pattern_height)
                    signal = 'strong_sell'
                else:
                    confidence = min(0.7, 0.3 + pattern_height)
                    signal = 'sell'

                return {
                    'pattern': 'double_top',
                    'signal': signal,
                    'confidence': confidence,
                    'breakout_level': valley_price,
                    'target': valley_price - (max(peak1_price, peak2_price) - valley_price),
                    'resistance': (peak1_price + peak2_price) / 2
                }

        return None

    def _detect_double_bottom(self, prices: pd.Series, troughs: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect double bottom pattern"""
        if len(troughs) < 2:
            return None

        recent_troughs = troughs[-5:] if len(troughs) > 5 else troughs

        for i in range(len(recent_troughs) - 1):
            trough1_idx, trough2_idx = recent_troughs[i], recent_troughs[i + 1]
            trough1_price, trough2_price = prices.iloc[trough1_idx], prices.iloc[trough2_idx]

            depth_diff = abs(trough1_price - trough2_price) / min(trough1_price, trough2_price)

            if depth_diff <= self.tolerance:
                peak_idx = prices.iloc[trough1_idx:trough2_idx].idxmax()
                peak_price = prices.iloc[peak_idx]

                pattern_height = (peak_price - min(trough1_price, trough2_price)) / peak_price
                current_price = prices.iloc[-1]

                if current_price > peak_price * 1.02:
                    confidence = min(0.9, 0.5 + pattern_height)
                    signal = 'strong_buy'
                else:
                    confidence = min(0.7, 0.3 + pattern_height)
                    signal = 'buy'

                return {
                    'pattern': 'double_bottom',
                    'signal': signal,
                    'confidence': confidence,
                    'breakout_level': peak_price,
                    'target': peak_price + (peak_price - min(trough1_price, trough2_price)),
                    'support': (trough1_price + trough2_price) / 2
                }

        return None

    def _detect_head_shoulders(self, prices: pd.Series, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect head and shoulders pattern (regular and inverse)"""
        # Regular head and shoulders
        if len(peaks) >= 3:
            recent_peaks = peaks[-5:] if len(peaks) > 5 else peaks

            for i in range(len(recent_peaks) - 2):
                left_shoulder = recent_peaks[i]
                head = recent_peaks[i + 1]
                right_shoulder = recent_peaks[i + 2]

                left_price = prices.iloc[left_shoulder]
                head_price = prices.iloc[head]
                right_price = prices.iloc[right_shoulder]

                # Head should be higher than shoulders
                if head_price > left_price and head_price > right_price:
                    shoulder_diff = abs(left_price - right_price) / max(left_price, right_price)

                    if shoulder_diff <= self.tolerance * 1.5:
                        # Find neckline
                        left_valley = prices.iloc[left_shoulder:head].idxmin()
                        right_valley = prices.iloc[head:right_shoulder].idxmin()
                        neckline = (prices.iloc[left_valley] + prices.iloc[right_valley]) / 2

                        pattern_height = (head_price - neckline) / neckline
                        confidence = min(0.85, 0.5 + pattern_height * 0.5)

                        return {
                            'pattern': 'head_and_shoulders',
                            'signal': 'sell',
                            'confidence': confidence,
                            'breakout_level': neckline,
                            'target': neckline - (head_price - neckline)
                        }

        # Inverse head and shoulders
        if len(troughs) >= 3:
            recent_troughs = troughs[-5:] if len(troughs) > 5 else troughs

            for i in range(len(recent_troughs) - 2):
                left_shoulder = recent_troughs[i]
                head = recent_troughs[i + 1]
                right_shoulder = recent_troughs[i + 2]

                left_price = prices.iloc[left_shoulder]
                head_price = prices.iloc[head]
                right_price = prices.iloc[right_shoulder]

                # Head should be lower than shoulders
                if head_price < left_price and head_price < right_price:
                    shoulder_diff = abs(left_price - right_price) / min(left_price, right_price)

                    if shoulder_diff <= self.tolerance * 1.5:
                        # Find neckline
                        left_peak = prices.iloc[left_shoulder:head].idxmax()
                        right_peak = prices.iloc[head:right_shoulder].idxmax()
                        neckline = (prices.iloc[left_peak] + prices.iloc[right_peak]) / 2

                        pattern_height = (neckline - head_price) / head_price
                        confidence = min(0.85, 0.5 + pattern_height * 0.5)

                        return {
                            'pattern': 'inverse_head_and_shoulders',
                            'signal': 'buy',
                            'confidence': confidence,
                            'breakout_level': neckline,
                            'target': neckline + (neckline - head_price)
                        }

        return None

    def _detect_triangle(self, prices: pd.Series, highs: pd.Series, lows: pd.Series) -> Optional[Dict[str, Any]]:
        """Detect triangle patterns"""
        if len(prices) < self.min_pattern_bars:
            return None

        recent_data = prices.tail(self.max_pattern_bars)
        recent_highs = highs.tail(self.max_pattern_bars)
        recent_lows = lows.tail(self.max_pattern_bars)

        x = np.arange(len(recent_data))

        # Calculate trend lines
        high_slope, high_intercept, high_r, _, _ = stats.linregress(x, recent_highs)
        low_slope, low_intercept, low_r, _, _ = stats.linregress(x, recent_lows)

        if abs(high_r) > 0.6 and abs(low_r) > 0.6:
            # Determine triangle type
            if abs(high_slope) < 0.0005 and low_slope > 0.001:
                # Ascending triangle (bullish)
                return {
                    'pattern': 'ascending_triangle',
                    'signal': 'buy',
                    'confidence': min(abs(low_r), 0.8),
                    'breakout_level': recent_highs.mean(),
                    'volume_confirmed': True  # Simplified
                }
            elif high_slope < -0.001 and abs(low_slope) < 0.0005:
                # Descending triangle (bearish)
                return {
                    'pattern': 'descending_triangle',
                    'signal': 'sell',
                    'confidence': min(abs(high_r), 0.8),
                    'breakout_level': recent_lows.mean(),
                    'volume_confirmed': True
                }
            elif high_slope < -0.001 and low_slope > 0.001:
                # Symmetric triangle (neutral, wait for breakout)
                return {
                    'pattern': 'symmetric_triangle',
                    'signal': 'hold',
                    'confidence': min(abs(high_r), abs(low_r), 0.7),
                    'resistance_trend': high_slope,
                    'support_trend': low_slope
                }

        return None

    def _detect_flag(self, prices: pd.Series, volumes: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
        """Detect flag patterns"""
        if len(prices) < 20:
            return None

        # Look for strong initial move
        lookback = 20
        recent_prices = prices.tail(lookback)

        # Calculate initial move
        initial_move = (recent_prices.iloc[10] - recent_prices.iloc[0]) / recent_prices.iloc[0]

        if abs(initial_move) > 0.08:  # 8% minimum move
            # Check consolidation
            consolidation = recent_prices.tail(10)
            volatility = consolidation.std() / consolidation.mean()

            if volatility < 0.03:  # Low volatility consolidation
                # Determine direction
                if initial_move > 0:
                    pattern = 'bull_flag'
                    signal = 'buy'
                else:
                    pattern = 'bear_flag'
                    signal = 'sell'

                # Volume confirmation
                volume_confirmed = True
                if volumes is not None and self.volume_confirmation:
                    recent_volumes = volumes.tail(10)
                    avg_volume = volumes.tail(20).mean()
                    volume_confirmed = recent_volumes.mean() < avg_volume * 0.7

                confidence = 0.75 if volume_confirmed else 0.6

                return {
                    'pattern': pattern,
                    'signal': signal,
                    'confidence': confidence,
                    'flagpole_move': abs(initial_move),
                    'volume_confirmed': volume_confirmed
                }

        return None

    def _detect_wedge(self, prices: pd.Series, highs: pd.Series, lows: pd.Series) -> Optional[Dict[str, Any]]:
        """Detect wedge patterns"""
        if len(prices) < self.min_pattern_bars:
            return None

        recent_highs = highs.tail(30)
        recent_lows = lows.tail(30)

        x = np.arange(len(recent_highs))

        # Calculate trend lines
        high_slope, _, high_r, _, _ = stats.linregress(x, recent_highs)
        low_slope, _, low_r, _, _ = stats.linregress(x, recent_lows)

        if abs(high_r) > 0.7 and abs(low_r) > 0.7:
            # Both trending in same direction = wedge
            if high_slope > 0.001 and low_slope > 0.001:
                # Rising wedge (bearish)
                return {
                    'pattern': 'rising_wedge',
                    'signal': 'sell',
                    'confidence': min(abs(high_r), abs(low_r), 0.75),
                    'breakout_expected': 'down'
                }
            elif high_slope < -0.001 and low_slope < -0.001:
                # Falling wedge (bullish)
                return {
                    'pattern': 'falling_wedge',
                    'signal': 'buy',
                    'confidence': min(abs(high_r), abs(low_r), 0.75),
                    'breakout_expected': 'up'
                }

        return None

    def _detect_channel(self, prices: pd.Series, highs: pd.Series, lows: pd.Series) -> Optional[Dict[str, Any]]:
        """Detect channel patterns"""
        if len(prices) < 20:
            return None

        recent_highs = highs.tail(30)
        recent_lows = lows.tail(30)
        recent_prices = prices.tail(30)

        # Calculate channel width
        channel_width = (recent_highs - recent_lows).mean()
        avg_price = recent_prices.mean()

        # Check if price is respecting channel
        touches_high = sum(recent_prices > recent_highs * 0.98)
        touches_low = sum(recent_prices < recent_lows * 1.02)

        if touches_high >= 2 and touches_low >= 2:
            # Valid channel
            current_price = prices.iloc[-1]
            position_in_channel = (current_price - recent_lows.iloc[-1]) / channel_width

            if position_in_channel > 0.8:
                signal = 'sell'
                confidence = 0.7
            elif position_in_channel < 0.2:
                signal = 'buy'
                confidence = 0.7
            else:
                signal = 'hold'
                confidence = 0.5

            return {
                'pattern': 'channel',
                'signal': signal,
                'confidence': confidence,
                'channel_top': recent_highs.iloc[-1],
                'channel_bottom': recent_lows.iloc[-1],
                'position_in_channel': position_in_channel
            }

        return None


# Export for compatibility
__all__ = ['PatternAgent']
