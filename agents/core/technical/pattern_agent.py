"""
Pattern Agent - Chart pattern recognition for trading signals.
Detects classic chart patterns: double top/bottom, head & shoulders, triangles, flags.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from scipy import stats
from scipy.signal import argrelextrema
from ..base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class PatternAgent(BaseAgent):
    """Agent that detects chart patterns and generates trading signals."""
    
    def __init__(
        self,
        name: str = "Pattern",
        min_pattern_bars: int = 10,
        max_pattern_bars: int = 50,
        tolerance: float = 0.02,
        volume_confirmation: bool = True
    ):
        """
        Initialize Pattern agent.
        
        Args:
            name: Agent name
            min_pattern_bars: Minimum bars for pattern detection
            max_pattern_bars: Maximum bars for pattern detection  
            tolerance: Price tolerance for pattern matching (2%)
            volume_confirmation: Require volume confirmation
        """
        super().__init__(name=name, agent_type="technical")
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.tolerance = tolerance
        self.volume_confirmation = volume_confirmation
        
    def find_peaks_troughs(self, prices: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks and troughs in price series."""
        try:
            # Find local maxima (peaks) and minima (troughs)
            peaks = argrelextrema(prices.values, np.greater, order=order)[0]
            troughs = argrelextrema(prices.values, np.less, order=order)[0]
            
            return peaks, troughs
        except Exception as e:
            logger.error(f"Peak/trough detection failed: {str(e)}")
            return np.array([]), np.array([])
    
    def detect_double_top(self, prices: pd.Series, peaks: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect double top pattern."""
        try:
            if len(peaks) < 2:
                return None
                
            # Look for two peaks with similar heights
            for i in range(len(peaks) - 1):
                peak1_idx, peak2_idx = peaks[i], peaks[i + 1]
                peak1_price, peak2_price = prices.iloc[peak1_idx], prices.iloc[peak2_idx]
                
                # Check if peaks are similar in height (within tolerance)
                height_diff = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
                
                if height_diff <= self.tolerance:
                    # Find the valley between peaks
                    valley_start, valley_end = peak1_idx, peak2_idx
                    valley_idx = prices.iloc[valley_start:valley_end].idxmin()
                    valley_price = prices.iloc[valley_idx]
                    
                    # Pattern strength based on height difference and valley depth
                    valley_depth = (min(peak1_price, peak2_price) - valley_price) / valley_price
                    
                    return {
                        'pattern': 'double_top',
                        'signal': 'sell',
                        'confidence': min(1.0, valley_depth * 2),
                        'resistance_level': (peak1_price + peak2_price) / 2,
                        'target': valley_price,
                        'peaks': [peak1_idx, peak2_idx],
                        'valley': valley_idx
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Double top detection failed: {str(e)}")
            return None
    
    def detect_double_bottom(self, prices: pd.Series, troughs: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect double bottom pattern."""
        try:
            if len(troughs) < 2:
                return None
                
            # Look for two troughs with similar lows
            for i in range(len(troughs) - 1):
                trough1_idx, trough2_idx = troughs[i], troughs[i + 1]
                trough1_price, trough2_price = prices.iloc[trough1_idx], prices.iloc[trough2_idx]
                
                # Check if troughs are similar in depth (within tolerance)
                depth_diff = abs(trough1_price - trough2_price) / min(trough1_price, trough2_price)
                
                if depth_diff <= self.tolerance:
                    # Find the peak between troughs
                    peak_start, peak_end = trough1_idx, trough2_idx
                    peak_idx = prices.iloc[peak_start:peak_end].idxmax()
                    peak_price = prices.iloc[peak_idx]
                    
                    # Pattern strength based on depth similarity and peak height
                    peak_height = (peak_price - max(trough1_price, trough2_price)) / peak_price
                    
                    return {
                        'pattern': 'double_bottom',
                        'signal': 'buy',
                        'confidence': min(1.0, peak_height * 2),
                        'support_level': (trough1_price + trough2_price) / 2,
                        'target': peak_price,
                        'troughs': [trough1_idx, trough2_idx],
                        'peak': peak_idx
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Double bottom detection failed: {str(e)}")
            return None
    
    def detect_head_shoulders(self, prices: pd.Series, peaks: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect head and shoulders pattern."""
        try:
            if len(peaks) < 3:
                return None
                
            # Look for three peaks: left shoulder, head, right shoulder
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                left_price = prices.iloc[left_shoulder]
                head_price = prices.iloc[head]
                right_price = prices.iloc[right_shoulder]
                
                # Head should be higher than both shoulders
                if head_price > left_price and head_price > right_price:
                    # Shoulders should be roughly equal height
                    shoulder_diff = abs(left_price - right_price) / max(left_price, right_price)
                    
                    if shoulder_diff <= self.tolerance:
                        # Find neckline (connecting the two valleys)
                        left_valley = prices.iloc[left_shoulder:head].idxmin()
                        right_valley = prices.iloc[head:right_shoulder].idxmin()
                        neckline = (prices.iloc[left_valley] + prices.iloc[right_valley]) / 2
                        
                        # Pattern strength
                        head_height = (head_price - neckline) / neckline
                        
                        return {
                            'pattern': 'head_shoulders',
                            'signal': 'sell',
                            'confidence': min(1.0, head_height),
                            'neckline': neckline,
                            'target': neckline - (head_price - neckline),
                            'peaks': [left_shoulder, head, right_shoulder]
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Head and shoulders detection failed: {str(e)}")
            return None
    
    def detect_triangle(self, prices: pd.Series, highs: pd.Series, lows: pd.Series) -> Optional[Dict[str, Any]]:
        """Detect triangle patterns (ascending, descending, symmetric)."""
        try:
            if len(prices) < self.min_pattern_bars:
                return None
            
            # Get recent highs and lows for trend analysis
            recent_data = prices.tail(self.max_pattern_bars)
            recent_highs = highs.tail(self.max_pattern_bars)
            recent_lows = lows.tail(self.max_pattern_bars)
            
            # Calculate trend lines for highs and lows
            x = np.arange(len(recent_data))
            
            # Trend of highs
            high_slope, high_intercept, high_r, _, _ = stats.linregress(x, recent_highs)
            
            # Trend of lows  
            low_slope, low_intercept, low_r, _, _ = stats.linregress(x, recent_lows)
            
            # Determine triangle type based on slope directions
            if abs(high_r) > 0.7 and abs(low_r) > 0.7:  # Strong correlation
                if high_slope < -0.001 and low_slope > 0.001:
                    # Converging lines - symmetric triangle
                    pattern_type = 'symmetric_triangle'
                    signal = 'hold'  # Breakout direction uncertain
                    confidence = min(abs(high_r), abs(low_r))
                elif high_slope > -0.001 and low_slope > 0.001:
                    # Ascending triangle
                    pattern_type = 'ascending_triangle'
                    signal = 'buy'
                    confidence = abs(low_r)
                elif high_slope < -0.001 and low_slope < 0.001:
                    # Descending triangle
                    pattern_type = 'descending_triangle'
                    signal = 'sell'
                    confidence = abs(high_r)
                else:
                    return None
                
                return {
                    'pattern': pattern_type,
                    'signal': signal,
                    'confidence': confidence * 0.8,  # Conservative confidence
                    'resistance_slope': high_slope,
                    'support_slope': low_slope,
                    'breakout_expected': True
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Triangle detection failed: {str(e)}")
            return None
    
    def detect_flag(self, prices: pd.Series, volume: Optional[pd.Series] = None) -> Optional[Dict[str, Any]]:
        """Detect flag pattern (consolidation after strong move)."""
        try:
            if len(prices) < self.min_pattern_bars:
                return None
            
            # Look for strong initial move (flagpole)
            lookback = min(20, len(prices) // 2)
            recent_prices = prices.tail(lookback)
            
            # Calculate the initial move strength
            initial_move = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            if abs(initial_move) < 0.05:  # Minimum 5% move for flagpole
                return None
            
            # Check for consolidation (flag)
            consolidation_period = min(10, lookback // 2)
            consolidation_prices = recent_prices.tail(consolidation_period)
            
            # Flag should have low volatility (tight range)
            volatility = consolidation_prices.std() / consolidation_prices.mean()
            
            if volatility < 0.02:  # Low volatility indicates consolidation
                # Determine flag direction
                flag_direction = 'bull_flag' if initial_move > 0 else 'bear_flag'
                signal = 'buy' if flag_direction == 'bull_flag' else 'sell'
                
                # Volume confirmation if available
                volume_confirmed = True
                if volume is not None and self.volume_confirmation:
                    recent_volume = volume.tail(consolidation_period)
                    avg_volume = volume.tail(lookback).mean()
                    volume_confirmed = recent_volume.mean() < avg_volume * 0.8  # Decreasing volume
                
                confidence = 0.7 if volume_confirmed else 0.5
                
                return {
                    'pattern': flag_direction,
                    'signal': signal,
                    'confidence': confidence,
                    'flagpole_move': initial_move,
                    'consolidation_volatility': volatility,
                    'volume_confirmed': volume_confirmed
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Flag detection failed: {str(e)}")
            return None
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and detect chart patterns."""
        try:
            if "close_prices" not in data:
                raise ValueError("Close prices not found in market data")
            
            prices = pd.Series(data["close_prices"])
            highs = pd.Series(data.get("high_prices", prices))
            lows = pd.Series(data.get("low_prices", prices))
            volume = pd.Series(data.get("volume", [])) if "volume" in data else None
            
            if len(prices) < self.min_pattern_bars:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Insufficient data for pattern detection"}
                }
            
            # Detect all patterns
            peaks, troughs = self.find_peaks_troughs(prices)
            
            patterns = []
            
            # Double patterns
            double_top = self.detect_double_top(prices, peaks)
            if double_top:
                patterns.append(double_top)
            
            double_bottom = self.detect_double_bottom(prices, troughs)
            if double_bottom:
                patterns.append(double_bottom)
            
            # Head and shoulders
            head_shoulders = self.detect_head_shoulders(prices, peaks)
            if head_shoulders:
                patterns.append(head_shoulders)
            
            # Triangle patterns
            triangle = self.detect_triangle(prices, highs, lows)
            if triangle:
                patterns.append(triangle)
            
            # Flag patterns
            flag = self.detect_flag(prices, volume)
            if flag:
                patterns.append(flag)
            
            # Select best pattern
            if patterns:
                best_pattern = max(patterns, key=lambda p: p['confidence'])
                
                return {
                    "action": best_pattern['signal'],
                    "confidence": best_pattern['confidence'],
                    "metadata": {
                        "detected_pattern": best_pattern,
                        "all_patterns": patterns,
                        "pattern_count": len(patterns)
                    }
                }
            
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"patterns_detected": 0}
            }
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 