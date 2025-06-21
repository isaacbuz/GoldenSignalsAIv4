"""
Hybrid Pattern Agent - Independent + Collaborative Pattern Recognition
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.common.hybrid_agent_base import HybridAgent
from agents.common.data_bus import SharedDataTypes

logger = logging.getLogger(__name__)

class HybridPatternAgent(HybridAgent):
    """
    Pattern Recognition Agent with dual signal generation
    
    Independent: Pure pattern detection (chart patterns, candlesticks)
    Collaborative: Patterns + volume confirmation, momentum alignment, support/resistance
    """
    
    def __init__(self, data_bus=None):
        super().__init__("HybridPatternAgent", data_bus)
        self.min_pattern_bars = 10
        self.pattern_lookback = 50
        
    def analyze_independent(self, symbol: str, data: Any = None) -> Dict[str, Any]:
        """Pure pattern analysis without external context"""
        try:
            # Fetch data if not provided
            if data is None:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="6mo")
            
            if len(data) < self.pattern_lookback:
                return self._create_signal("HOLD", 0.0, "Insufficient data for pattern analysis")
            
            # Detect various patterns
            patterns_found = []
            
            # Chart patterns
            chart_patterns = self._detect_chart_patterns(data)
            if chart_patterns:
                patterns_found.extend(chart_patterns)
            
            # Candlestick patterns
            candle_patterns = self._detect_candlestick_patterns(data)
            if candle_patterns:
                patterns_found.extend(candle_patterns)
            
            # Support/Resistance levels
            sr_levels = self._identify_support_resistance(data)
            
            # Generate signal from patterns
            action = "HOLD"
            confidence = 0.0
            reasoning = []
            
            if patterns_found:
                # Aggregate pattern signals
                bullish_patterns = [p for p in patterns_found if p['bias'] == 'bullish']
                bearish_patterns = [p for p in patterns_found if p['bias'] == 'bearish']
                
                if bullish_patterns and not bearish_patterns:
                    action = "BUY"
                    confidence = max(p['confidence'] for p in bullish_patterns)
                    pattern_names = [p['name'] for p in bullish_patterns]
                    reasoning.append(f"Bullish patterns: {', '.join(pattern_names)}")
                    
                elif bearish_patterns and not bullish_patterns:
                    action = "SELL"
                    confidence = max(p['confidence'] for p in bearish_patterns)
                    pattern_names = [p['name'] for p in bearish_patterns]
                    reasoning.append(f"Bearish patterns: {', '.join(pattern_names)}")
                    
                elif bullish_patterns and bearish_patterns:
                    # Mixed signals - use highest confidence
                    max_bull = max(bullish_patterns, key=lambda x: x['confidence'])
                    max_bear = max(bearish_patterns, key=lambda x: x['confidence'])
                    
                    if max_bull['confidence'] > max_bear['confidence']:
                        action = "BUY"
                        confidence = max_bull['confidence'] * 0.8  # Reduce for mixed signals
                        reasoning.append(f"Bullish pattern ({max_bull['name']}) dominates")
                    else:
                        action = "SELL"
                        confidence = max_bear['confidence'] * 0.8
                        reasoning.append(f"Bearish pattern ({max_bear['name']}) dominates")
            
            # Check price vs support/resistance
            current_price = data['Close'].iloc[-1]
            if sr_levels['support_levels'] and action == "HOLD":
                nearest_support = min(sr_levels['support_levels'], 
                                    key=lambda x: abs(x - current_price))
                if abs(current_price - nearest_support) / current_price < 0.01:
                    action = "BUY"
                    confidence = 0.5
                    reasoning.append("Price at support level")
            
            # Share insights
            if self.data_bus:
                self._share_pattern_insights(symbol, patterns_found, sr_levels)
            
            confidence = min(confidence, 0.85)  # Cap independent confidence
            
            return self._create_signal(
                action,
                confidence,
                " | ".join(reasoning) if reasoning else "No significant patterns detected",
                {
                    'patterns': patterns_found,
                    'support_resistance': sr_levels,
                    'pattern_count': len(patterns_found)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in pattern independent analysis: {e}")
            return self._create_signal("HOLD", 0.0, f"Error: {str(e)}")
    
    def analyze_collaborative(self, symbol: str, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pattern analysis enhanced with market context"""
        try:
            # Start with independent analysis
            base_signal = self.analyze_independent(symbol, data)
            
            # Get context
            volume_data = context['data'].get(SharedDataTypes.VOLUME_SPIKES, {})
            volume_profile = context['data'].get(SharedDataTypes.VOLUME_PROFILE, {})
            momentum_state = context['data'].get(SharedDataTypes.MOMENTUM_STATE, {})
            overbought_oversold = context['data'].get(SharedDataTypes.OVERBOUGHT_OVERSOLD, {})
            
            # Enhanced analysis
            action = base_signal['action']
            confidence = base_signal['confidence']
            reasons = [base_signal['metadata']['reasoning']]
            adjustments = []
            
            patterns = base_signal['metadata']['indicators'].get('patterns', [])
            
            # Volume confirmation for patterns
            if volume_data and patterns:
                for agent, vol_info in volume_data.items():
                    spike_data = vol_info['data']
                    
                    # Check if volume confirms pattern
                    for pattern in patterns:
                        if pattern['bias'] == 'bullish' and spike_data.get('spike_type') == 'bullish_spike':
                            confidence += 0.15
                            adjustments.append(f"Volume confirms {pattern['name']}")
                            break
                        elif pattern['bias'] == 'bearish' and spike_data.get('spike_type') == 'bearish_spike':
                            confidence += 0.15
                            adjustments.append(f"Volume confirms {pattern['name']}")
                            break
                        elif pattern['type'] == 'reversal' and spike_data.get('spike_type') == 'absorption':
                            confidence += 0.1
                            adjustments.append("Absorption at reversal pattern")
                            break
            
            # Volume profile context
            if volume_profile and data is not None:
                current_price = data['Close'].iloc[-1]
                
                for agent, vp_data in volume_profile.items():
                    poc = vp_data['data'].get('poc', 0)
                    
                    if poc > 0:
                        # Patterns above/below POC
                        if action == "BUY" and current_price < poc:
                            confidence += 0.1
                            adjustments.append("Pattern below POC - good risk/reward")
                        elif action == "SELL" and current_price > poc:
                            confidence += 0.1
                            adjustments.append("Pattern above POC - good risk/reward")
            
            # Momentum alignment
            if momentum_state:
                for agent, mom_data in momentum_state.items():
                    momentum = mom_data['data'].get('state', '')
                    
                    if momentum == 'bullish' and action == "BUY":
                        confidence += 0.1
                        adjustments.append("Momentum aligns with bullish pattern")
                    elif momentum == 'bearish' and action == "SELL":
                        confidence += 0.1
                        adjustments.append("Momentum aligns with bearish pattern")
                    elif (momentum == 'bullish' and action == "SELL") or \
                         (momentum == 'bearish' and action == "BUY"):
                        confidence *= 0.8
                        adjustments.append("Momentum contradicts pattern - caution")
            
            # Overbought/Oversold context
            if overbought_oversold:
                for agent, ob_os_data in overbought_oversold.items():
                    state = ob_os_data['data'].get('state', '')
                    
                    # Reversal patterns at extremes
                    reversal_patterns = [p for p in patterns if p['type'] == 'reversal']
                    if reversal_patterns:
                        if state == 'overbought' and action == "SELL":
                            confidence += 0.1
                            adjustments.append("Reversal pattern at overbought level")
                        elif state == 'oversold' and action == "BUY":
                            confidence += 0.1
                            adjustments.append("Reversal pattern at oversold level")
                    
                    # Continuation patterns
                    continuation_patterns = [p for p in patterns if p['type'] == 'continuation']
                    if continuation_patterns and state == 'neutral':
                        confidence += 0.05
                        adjustments.append("Continuation pattern in neutral zone")
            
            # Pattern quality based on context
            if len(adjustments) >= 2:
                confidence += 0.05
                adjustments.append("Multiple confirmations")
            elif len(adjustments) == 0 and patterns:
                confidence *= 0.9
                adjustments.append("Pattern lacks confirmation")
            
            # Cap collaborative confidence
            confidence = min(confidence, 0.95)
            
            # Build enhanced reasoning
            if adjustments:
                reasons.extend(adjustments)
            
            return self._create_signal(
                action,
                confidence,
                " | ".join(reasons),
                {
                    **base_signal['metadata']['indicators'],
                    'context_confirmations': len(adjustments),
                    'collaborative_adjustments': adjustments
                }
            )
            
        except Exception as e:
            logger.error(f"Error in pattern collaborative analysis: {e}")
            return base_signal
    
    def _detect_chart_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect major chart patterns"""
        patterns = []
        
        # Head and Shoulders
        hs_pattern = self._detect_head_shoulders(data)
        if hs_pattern:
            patterns.append(hs_pattern)
        
        # Double Top/Bottom
        double_pattern = self._detect_double_pattern(data)
        if double_pattern:
            patterns.append(double_pattern)
        
        # Triangles
        triangle = self._detect_triangle(data)
        if triangle:
            patterns.append(triangle)
        
        # Flag/Pennant
        flag = self._detect_flag(data)
        if flag:
            patterns.append(flag)
        
        return patterns
    
    def _detect_head_shoulders(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect head and shoulders pattern"""
        try:
            highs = data['High'].rolling(5).max()
            lows = data['Low'].rolling(5).min()
            
            # Look for pattern in last 30 bars
            recent_highs = highs.tail(30)
            
            # Find peaks
            peaks = []
            for i in range(1, len(recent_highs)-1):
                if recent_highs.iloc[i] > recent_highs.iloc[i-1] and \
                   recent_highs.iloc[i] > recent_highs.iloc[i+1]:
                    peaks.append((i, recent_highs.iloc[i]))
            
            # Need at least 3 peaks for H&S
            if len(peaks) >= 3:
                # Check if middle peak is highest (head)
                peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
                if peaks_sorted[0][0] > peaks[0][0] and peaks_sorted[0][0] < peaks[-1][0]:
                    # Found potential H&S
                    left_shoulder = peaks[0][1]
                    head = peaks_sorted[0][1]
                    right_shoulder = peaks[-1][1]
                    
                    # Check symmetry
                    if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05:
                        neckline = lows.iloc[peaks[0][0]:peaks[-1][0]].min()
                        current_price = data['Close'].iloc[-1]
                        
                        if current_price < neckline:
                            return {
                                'name': 'Head and Shoulders',
                                'type': 'reversal',
                                'bias': 'bearish',
                                'confidence': 0.75,
                                'target': neckline - (head - neckline)
                            }
            
            # Inverse H&S (similar logic with lows)
            # ... (simplified for brevity)
            
        except Exception as e:
            logger.error(f"Error detecting H&S: {e}")
        
        return None
    
    def _detect_double_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect double top/bottom patterns"""
        try:
            highs = data['High'].tail(30)
            lows = data['Low'].tail(30)
            closes = data['Close'].tail(30)
            
            # Double top
            max_high = highs.max()
            high_indices = highs[highs > max_high * 0.98].index
            
            if len(high_indices) >= 2:
                first_peak = high_indices[0]
                last_peak = high_indices[-1]
                
                if last_peak - first_peak >= 10:  # Adequate separation
                    valley = lows.loc[first_peak:last_peak].min()
                    if closes.iloc[-1] < valley:
                        return {
                            'name': 'Double Top',
                            'type': 'reversal',
                            'bias': 'bearish',
                            'confidence': 0.7,
                            'target': valley - (max_high - valley)
                        }
            
            # Double bottom
            min_low = lows.min()
            low_indices = lows[lows < min_low * 1.02].index
            
            if len(low_indices) >= 2:
                first_bottom = low_indices[0]
                last_bottom = low_indices[-1]
                
                if last_bottom - first_bottom >= 10:
                    peak = highs.loc[first_bottom:last_bottom].max()
                    if closes.iloc[-1] > peak:
                        return {
                            'name': 'Double Bottom',
                            'type': 'reversal',
                            'bias': 'bullish',
                            'confidence': 0.7,
                            'target': peak + (peak - min_low)
                        }
            
        except Exception as e:
            logger.error(f"Error detecting double pattern: {e}")
        
        return None
    
    def _detect_triangle(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect triangle patterns"""
        try:
            recent_data = data.tail(20)
            highs = recent_data['High']
            lows = recent_data['Low']
            
            # Calculate trendlines
            high_slope = (highs.iloc[-1] - highs.iloc[0]) / len(highs)
            low_slope = (lows.iloc[-1] - lows.iloc[0]) / len(lows)
            
            # Ascending triangle
            if abs(high_slope) < 0.001 and low_slope > 0:
                return {
                    'name': 'Ascending Triangle',
                    'type': 'continuation',
                    'bias': 'bullish',
                    'confidence': 0.65,
                    'target': highs.mean() + (highs.mean() - lows.mean())
                }
            
            # Descending triangle
            elif high_slope < 0 and abs(low_slope) < 0.001:
                return {
                    'name': 'Descending Triangle',
                    'type': 'continuation',
                    'bias': 'bearish',
                    'confidence': 0.65,
                    'target': lows.mean() - (highs.mean() - lows.mean())
                }
            
            # Symmetrical triangle
            elif abs(high_slope + low_slope) < 0.001:
                # Bias depends on prior trend
                prior_trend = data['Close'].iloc[-30] - data['Close'].iloc[-60]
                bias = 'bullish' if prior_trend > 0 else 'bearish'
                
                return {
                    'name': 'Symmetrical Triangle',
                    'type': 'continuation',
                    'bias': bias,
                    'confidence': 0.6,
                    'target': None
                }
            
        except Exception as e:
            logger.error(f"Error detecting triangle: {e}")
        
        return None
    
    def _detect_flag(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect flag and pennant patterns"""
        try:
            # Look for strong move followed by consolidation
            recent_data = data.tail(15)
            prior_data = data.iloc[-30:-15]
            
            # Calculate prior move
            prior_move = (prior_data['Close'].iloc[-1] - prior_data['Close'].iloc[0]) / prior_data['Close'].iloc[0]
            
            # Calculate consolidation range
            consolidation_range = (recent_data['High'].max() - recent_data['Low'].min()) / recent_data['Close'].mean()
            
            # Bull flag
            if prior_move > 0.05 and consolidation_range < 0.02:
                return {
                    'name': 'Bull Flag',
                    'type': 'continuation',
                    'bias': 'bullish',
                    'confidence': 0.7,
                    'target': recent_data['Close'].iloc[-1] + (prior_data['Close'].iloc[-1] - prior_data['Close'].iloc[0])
                }
            
            # Bear flag
            elif prior_move < -0.05 and consolidation_range < 0.02:
                return {
                    'name': 'Bear Flag',
                    'type': 'continuation',
                    'bias': 'bearish',
                    'confidence': 0.7,
                    'target': recent_data['Close'].iloc[-1] - (prior_data['Close'].iloc[0] - prior_data['Close'].iloc[-1])
                }
            
        except Exception as e:
            logger.error(f"Error detecting flag: {e}")
        
        return None
    
    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect candlestick patterns"""
        patterns = []
        
        # Get recent candles
        recent = data.tail(3)
        if len(recent) < 3:
            return patterns
        
        # Current and previous candles
        curr = recent.iloc[-1]
        prev = recent.iloc[-2]
        prev2 = recent.iloc[-3]
        
        # Hammer/Hanging Man
        body = abs(curr['Close'] - curr['Open'])
        lower_shadow = min(curr['Open'], curr['Close']) - curr['Low']
        upper_shadow = curr['High'] - max(curr['Open'], curr['Close'])
        
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            if prev['Close'] < prev['Open']:  # After downtrend
                patterns.append({
                    'name': 'Hammer',
                    'type': 'reversal',
                    'bias': 'bullish',
                    'confidence': 0.65
                })
            else:  # After uptrend
                patterns.append({
                    'name': 'Hanging Man',
                    'type': 'reversal',
                    'bias': 'bearish',
                    'confidence': 0.6
                })
        
        # Engulfing patterns
        if prev['Close'] < prev['Open'] and curr['Close'] > curr['Open']:
            if curr['Close'] > prev['Open'] and curr['Open'] < prev['Close']:
                patterns.append({
                    'name': 'Bullish Engulfing',
                    'type': 'reversal',
                    'bias': 'bullish',
                    'confidence': 0.7
                })
        
        elif prev['Close'] > prev['Open'] and curr['Close'] < curr['Open']:
            if curr['Close'] < prev['Open'] and curr['Open'] > prev['Close']:
                patterns.append({
                    'name': 'Bearish Engulfing',
                    'type': 'reversal',
                    'bias': 'bearish',
                    'confidence': 0.7
                })
        
        # Doji
        if body < (curr['High'] - curr['Low']) * 0.1:
            patterns.append({
                'name': 'Doji',
                'type': 'reversal',
                'bias': 'neutral',
                'confidence': 0.5
            })
        
        return patterns
    
    def _identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify key support and resistance levels"""
        levels = {'support_levels': [], 'resistance_levels': []}
        
        try:
            # Use pivot points
            highs = data['High'].tail(50)
            lows = data['Low'].tail(50)
            
            # Find local minima (support)
            for i in range(2, len(lows)-2):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
                   lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
                    levels['support_levels'].append(float(lows.iloc[i]))
            
            # Find local maxima (resistance)
            for i in range(2, len(highs)-2):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
                   highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
                    levels['resistance_levels'].append(float(highs.iloc[i]))
            
            # Keep only significant levels
            levels['support_levels'] = sorted(set(levels['support_levels']))[-3:]
            levels['resistance_levels'] = sorted(set(levels['resistance_levels']))[:3]
            
        except Exception as e:
            logger.error(f"Error identifying S/R levels: {e}")
        
        return levels
    
    def _share_pattern_insights(self, symbol: str, patterns: List[Dict], sr_levels: Dict):
        """Share pattern insights via data bus"""
        if not self.data_bus:
            return
        
        # Share pattern detection
        if patterns:
            bullish = any(p['bias'] == 'bullish' for p in patterns)
            bearish = any(p['bias'] == 'bearish' for p in patterns)
            
            self.data_bus.publish(
                self.name,
                symbol,
                SharedDataTypes.PRICE_PATTERNS,
                {
                    'bullish_pattern': bullish,
                    'bearish_pattern': bearish,
                    'pattern_names': [p['name'] for p in patterns],
                    'strongest_pattern': max(patterns, key=lambda x: x['confidence']) if patterns else None
                }
            )
        
        # Share support/resistance
        if sr_levels['support_levels'] or sr_levels['resistance_levels']:
            self.data_bus.publish(
                self.name,
                symbol,
                SharedDataTypes.SUPPORT_RESISTANCE,
                sr_levels
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
        """Specify what context pattern agent needs"""
        if self.data_bus:
            return self.data_bus.get_context(symbol, [
                SharedDataTypes.VOLUME_SPIKES,
                SharedDataTypes.VOLUME_PROFILE,
                SharedDataTypes.MOMENTUM_STATE,
                SharedDataTypes.OVERBOUGHT_OVERSOLD
            ])
        return {} 