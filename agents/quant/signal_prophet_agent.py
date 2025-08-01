"""
AI Signal Prophet Agent - The Trading Prophet
Generates high-confidence signals with visual chart execution guidance
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from agents.common.base.base_agent import BaseAgent
from agents.common.models.signal import Signal, SignalType
from agents.common.utils.logger import get_logger

logger = get_logger(__name__)

class SignalConfidence(Enum):
    """Signal confidence levels"""
    ULTRA_HIGH = "ultra_high"  # 90-100%
    HIGH = "high"              # 80-90%
    MEDIUM = "medium"          # 70-80%
    LOW = "low"                # 60-70%

@dataclass
class ProphetSignal:
    """Enhanced signal with prophet-specific attributes"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    entry_price: float
    stop_loss: float
    take_profits: List[float]  # Multiple TP levels
    confidence: float
    confidence_level: SignalConfidence
    pattern: str
    timeframe: str

    # Visual execution instructions
    chart_annotations: List[Dict[str, Any]]
    entry_zone: Tuple[float, float]  # Entry range
    risk_reward_ratio: float

    # Prophet insights
    market_context: str
    key_levels: Dict[str, float]
    trade_narrative: str
    risk_factors: List[str]

    # Timing
    signal_time: datetime
    expiry_time: datetime
    holding_period: str

class AISignalProphet(BaseAgent):
    """
    The Trading Prophet - AI agent that generates high-confidence signals
    with complete visual execution guidance
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "AI Signal Prophet"
        self.description = "Prophetic signal generation with visual execution"

        # Prophet configuration
        self.min_confidence = config.get('min_confidence', 70)
        self.max_concurrent_signals = config.get('max_concurrent_signals', 5)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)

        # Pattern recognition
        self.patterns = {
            'breakout': self.detect_breakout_pattern,
            'reversal': self.detect_reversal_pattern,
            'continuation': self.detect_continuation_pattern,
            'harmonic': self.detect_harmonic_pattern,
        }

        # Active signals tracking
        self.active_signals = []
        self.signal_history = []

    async def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Main prophet analysis - generates signals with visual guidance
        """
        try:
            # 1. Market context analysis
            context = await self._analyze_market_context(market_data)

            # 2. Multi-timeframe analysis
            mtf_analysis = await self._multi_timeframe_analysis(market_data)

            # 3. Pattern detection across all types
            patterns = await self._detect_all_patterns(market_data)

            # 4. Generate prophet signals
            prophet_signals = await self._generate_prophet_signals(
                patterns, context, mtf_analysis, market_data
            )

            # 5. Filter by confidence and risk
            filtered_signals = self._filter_signals(prophet_signals)

            # 6. Add visual execution instructions
            visual_signals = await self._add_visual_instructions(filtered_signals)

            # 7. Convert to standard signals
            return self._convert_to_standard_signals(visual_signals)

        except Exception as e:
            logger.error(f"Prophet analysis error: {e}")
            return []

    async def _analyze_market_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze overall market context
        """
        df = pd.DataFrame(market_data['candles'])

        # Market regime
        volatility = df['close'].pct_change().std() * np.sqrt(252)
        trend_strength = self._calculate_trend_strength(df)

        # Key levels
        support_levels = self._find_support_levels(df)
        resistance_levels = self._find_resistance_levels(df)

        # Market structure
        higher_highs = self._count_higher_highs(df)
        lower_lows = self._count_lower_lows(df)

        context = {
            'regime': self._determine_regime(volatility, trend_strength),
            'volatility': volatility,
            'trend_strength': trend_strength,
            'trend_direction': 'bullish' if trend_strength > 0 else 'bearish',
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'structure': {
                'higher_highs': higher_highs,
                'lower_lows': lower_lows,
                'trend_intact': higher_highs > lower_lows
            }
        }

        return context

    async def _multi_timeframe_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze multiple timeframes for confluence
        """
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        mtf_signals = {}

        for tf in timeframes:
            # Simulate different timeframe data (in production, fetch actual data)
            tf_data = self._resample_data(market_data, tf)

            # Analyze each timeframe
            mtf_signals[tf] = {
                'trend': self._get_trend(tf_data),
                'momentum': self._get_momentum(tf_data),
                'support': self._get_nearest_support(tf_data),
                'resistance': self._get_nearest_resistance(tf_data),
            }

        # Calculate confluence score
        confluence = self._calculate_confluence(mtf_signals)

        return {
            'timeframes': mtf_signals,
            'confluence_score': confluence,
            'primary_trend': self._get_primary_trend(mtf_signals),
            'key_timeframe': self._identify_key_timeframe(mtf_signals)
        }

    async def _detect_all_patterns(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect all pattern types
        """
        patterns = []

        for pattern_type, detector in self.patterns.items():
            detected = await detector(market_data)
            patterns.extend(detected)

        # Sort by confidence
        patterns.sort(key=lambda x: x['confidence'], reverse=True)

        return patterns

    async def detect_breakout_pattern(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect breakout patterns
        """
        df = pd.DataFrame(market_data['candles'])
        patterns = []

        # Triangle breakout
        if self._is_triangle_pattern(df):
            current_price = df['close'].iloc[-1]
            breakout_level = self._calculate_triangle_breakout(df)

            if abs(current_price - breakout_level) / breakout_level < 0.01:  # Near breakout
                patterns.append({
                    'type': 'triangle_breakout',
                    'confidence': 85,
                    'entry': breakout_level * 1.002,  # Entry above breakout
                    'stop': breakout_level * 0.98,
                    'targets': [
                        breakout_level * 1.02,
                        breakout_level * 1.04,
                        breakout_level * 1.06
                    ],
                    'pattern_data': {
                        'apex': breakout_level,
                        'formation_bars': 20,
                        'volume_confirmation': self._check_volume_expansion(df)
                    }
                })

        # Range breakout
        range_data = self._identify_range(df)
        if range_data['is_range']:
            if self._is_breaking_range(df, range_data):
                patterns.append({
                    'type': 'range_breakout',
                    'confidence': 80,
                    'entry': range_data['resistance'] * 1.001,
                    'stop': range_data['support'],
                    'targets': [
                        range_data['resistance'] + (range_data['resistance'] - range_data['support']) * 0.5,
                        range_data['resistance'] + (range_data['resistance'] - range_data['support']) * 1.0,
                        range_data['resistance'] + (range_data['resistance'] - range_data['support']) * 1.5,
                    ],
                    'pattern_data': range_data
                })

        return patterns

    async def detect_reversal_pattern(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect reversal patterns
        """
        df = pd.DataFrame(market_data['candles'])
        patterns = []

        # Double bottom/top
        if self._is_double_bottom(df):
            neckline = self._find_neckline(df, 'double_bottom')
            patterns.append({
                'type': 'double_bottom',
                'confidence': 82,
                'entry': neckline * 1.001,
                'stop': df['low'].min() * 0.99,
                'targets': [
                    neckline + (neckline - df['low'].min()) * 0.5,
                    neckline + (neckline - df['low'].min()) * 1.0,
                    neckline + (neckline - df['low'].min()) * 1.5,
                ],
                'pattern_data': {
                    'bottom1': df['low'].iloc[-20:-10].min(),
                    'bottom2': df['low'].iloc[-10:].min(),
                    'neckline': neckline
                }
            })

        # Head and shoulders
        if self._is_head_and_shoulders(df):
            neckline = self._find_neckline(df, 'h&s')
            patterns.append({
                'type': 'head_and_shoulders',
                'confidence': 88,
                'entry': neckline * 0.999,
                'stop': df['high'].max() * 1.01,
                'targets': [
                    neckline - (df['high'].max() - neckline) * 0.5,
                    neckline - (df['high'].max() - neckline) * 1.0,
                    neckline - (df['high'].max() - neckline) * 1.5,
                ],
                'pattern_data': {
                    'left_shoulder': self._find_left_shoulder(df),
                    'head': df['high'].max(),
                    'right_shoulder': self._find_right_shoulder(df),
                    'neckline': neckline
                }
            })

        return patterns

    async def detect_continuation_pattern(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect continuation patterns
        """
        df = pd.DataFrame(market_data['candles'])
        patterns = []

        # Bull/Bear flag
        if self._is_flag_pattern(df):
            flag_data = self._analyze_flag(df)
            patterns.append({
                'type': f"{flag_data['direction']}_flag",
                'confidence': 78,
                'entry': flag_data['breakout_level'],
                'stop': flag_data['flag_low'] if flag_data['direction'] == 'bull' else flag_data['flag_high'],
                'targets': flag_data['targets'],
                'pattern_data': flag_data
            })

        return patterns

    async def detect_harmonic_pattern(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect harmonic patterns (Gartley, Butterfly, etc.)
        """
        df = pd.DataFrame(market_data['candles'])
        patterns = []

        # Simplified harmonic detection
        if self._is_gartley_pattern(df):
            completion_point = self._calculate_gartley_completion(df)
            patterns.append({
                'type': 'gartley',
                'confidence': 75,
                'entry': completion_point['D'],
                'stop': completion_point['D'] * 0.97,
                'targets': [
                    completion_point['D'] * 1.02,
                    completion_point['D'] * 1.04,
                    completion_point['D'] * 1.06,
                ],
                'pattern_data': completion_point
            })

        return patterns

    async def _generate_prophet_signals(self, patterns: List[Dict],
                                      context: Dict, mtf: Dict,
                                      market_data: Dict) -> List[ProphetSignal]:
        """
        Generate prophet signals from detected patterns
        """
        prophet_signals = []

        for pattern in patterns[:self.max_concurrent_signals]:  # Limit signals
            # Calculate enhanced confidence
            confidence = self._calculate_prophet_confidence(pattern, context, mtf)

            if confidence < self.min_confidence:
                continue

            # Generate signal
            signal = ProphetSignal(
                symbol=market_data['symbol'],
                action='BUY' if pattern['entry'] > market_data['candles'][-1]['close'] else 'SELL',
                entry_price=pattern['entry'],
                stop_loss=pattern['stop'],
                take_profits=pattern['targets'],
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                pattern=pattern['type'],
                timeframe=market_data.get('timeframe', '5m'),

                # Visual instructions
                chart_annotations=self._create_chart_annotations(pattern),
                entry_zone=(pattern['entry'] * 0.999, pattern['entry'] * 1.001),
                risk_reward_ratio=self._calculate_risk_reward(pattern),

                # Prophet insights
                market_context=self._generate_market_narrative(context),
                key_levels={
                    'support': context['support_levels'][0] if context['support_levels'] else 0,
                    'resistance': context['resistance_levels'][0] if context['resistance_levels'] else 0,
                    'pivot': (pattern['entry'] + pattern['stop']) / 2
                },
                trade_narrative=self._generate_trade_narrative(pattern, context),
                risk_factors=self._identify_risk_factors(pattern, context, mtf),

                # Timing
                signal_time=datetime.now(),
                expiry_time=datetime.now() + timedelta(hours=4),
                holding_period=self._estimate_holding_period(pattern)
            )

            prophet_signals.append(signal)

        return prophet_signals

    def _calculate_prophet_confidence(self, pattern: Dict, context: Dict, mtf: Dict) -> float:
        """
        Calculate enhanced confidence score
        """
        base_confidence = pattern['confidence']

        # Context bonus
        if context['regime'] == 'trending' and context['structure']['trend_intact']:
            base_confidence += 5

        # Multi-timeframe confluence bonus
        if mtf['confluence_score'] > 0.7:
            base_confidence += 10

        # Volume confirmation
        if pattern.get('pattern_data', {}).get('volume_confirmation', False):
            base_confidence += 5

        # Cap at 95
        return min(base_confidence, 95)

    def _create_chart_annotations(self, pattern: Dict) -> List[Dict[str, Any]]:
        """
        Create visual chart annotations for the signal
        """
        annotations = []

        # Entry point
        annotations.append({
            'type': 'entry_marker',
            'price': pattern['entry'],
            'text': f"Entry @ {pattern['entry']:.2f}",
            'color': 'green' if pattern['entry'] > pattern['stop'] else 'red',
            'style': 'arrow'
        })

        # Stop loss line
        annotations.append({
            'type': 'horizontal_line',
            'price': pattern['stop'],
            'text': f"Stop Loss @ {pattern['stop']:.2f}",
            'color': 'red',
            'style': 'dashed',
            'width': 2
        })

        # Take profit levels
        for i, tp in enumerate(pattern['targets']):
            annotations.append({
                'type': 'horizontal_line',
                'price': tp,
                'text': f"TP{i+1} @ {tp:.2f}",
                'color': 'green',
                'style': 'dotted',
                'width': 1
            })

        # Pattern visualization
        if 'pattern_data' in pattern:
            annotations.extend(self._create_pattern_annotations(pattern))

        return annotations

    def _generate_trade_narrative(self, pattern: Dict, context: Dict) -> str:
        """
        Generate human-readable trade explanation
        """
        direction = "bullish" if pattern['entry'] > pattern['stop'] else "bearish"

        narrative = f"AI Prophet has identified a high-confidence {pattern['type']} pattern. "
        narrative += f"The market is showing {direction} momentum with {context['regime']} conditions. "
        narrative += f"Entry is recommended at {pattern['entry']:.2f} with a stop at {pattern['stop']:.2f}. "
        narrative += f"Risk/Reward ratio is {self._calculate_risk_reward(pattern):.2f}:1. "
        narrative += f"Targets are set at {', '.join([f'{tp:.2f}' for tp in pattern['targets']])}."

        return narrative

    def _identify_risk_factors(self, pattern: Dict, context: Dict, mtf: Dict) -> List[str]:
        """
        Identify potential risks for the trade
        """
        risks = []

        # Volatility risk
        if context['volatility'] > 0.3:
            risks.append("High volatility - consider reducing position size")

        # Counter-trend risk
        if mtf['primary_trend'] != ('bullish' if pattern['entry'] > pattern['stop'] else 'bearish'):
            risks.append("Trading against primary trend - use tight stops")

        # Support/Resistance proximity
        if self._near_major_level(pattern['entry'], context):
            risks.append("Entry near major level - expect potential rejection")

        # Time-based risks
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour > 16:
            risks.append("Outside regular trading hours - lower liquidity")

        return risks

    def _filter_signals(self, signals: List[ProphetSignal]) -> List[ProphetSignal]:
        """
        Filter signals by quality and risk management
        """
        # Remove low confidence
        filtered = [s for s in signals if s.confidence >= self.min_confidence]

        # Remove conflicting signals
        filtered = self._remove_conflicts(filtered)

        # Apply position limits
        if len(self.active_signals) >= self.max_concurrent_signals:
            return []

        return filtered[:self.max_concurrent_signals - len(self.active_signals)]

    async def _add_visual_instructions(self, signals: List[ProphetSignal]) -> List[ProphetSignal]:
        """
        Add detailed visual execution instructions
        """
        for signal in signals:
            # Add execution steps
            signal.execution_steps = [
                f"1. Wait for price to enter zone {signal.entry_zone[0]:.2f} - {signal.entry_zone[1]:.2f}",
                f"2. Place stop loss at {signal.stop_loss:.2f}",
                f"3. Set TP1 at {signal.take_profits[0]:.2f} (50% position)",
                f"4. Set TP2 at {signal.take_profits[1]:.2f} (30% position)",
                f"5. Set TP3 at {signal.take_profits[2]:.2f} (20% position)",
                f"6. Move stop to breakeven after TP1 hit"
            ]

            # Add chart setup instructions
            signal.chart_setup = {
                'timeframe': signal.timeframe,
                'indicators': ['EMA 20', 'EMA 50', 'RSI', 'Volume'],
                'draw_pattern': True,
                'mark_levels': True
            }

        return signals

    def _convert_to_standard_signals(self, prophet_signals: List[ProphetSignal]) -> List[Signal]:
        """
        Convert prophet signals to standard format
        """
        standard_signals = []

        for ps in prophet_signals:
            signal = Signal(
                symbol=ps.symbol,
                signal_type=SignalType.BUY if ps.action == 'BUY' else SignalType.SELL,
                confidence=ps.confidence,
                entry_price=ps.entry_price,
                stop_loss=ps.stop_loss,
                take_profit=ps.take_profits[0],  # Primary target
                metadata={
                    'prophet_signal': True,
                    'pattern': ps.pattern,
                    'all_targets': ps.take_profits,
                    'risk_reward': ps.risk_reward_ratio,
                    'narrative': ps.trade_narrative,
                    'risks': ps.risk_factors,
                    'chart_annotations': ps.chart_annotations,
                    'market_context': ps.market_context,
                    'key_levels': ps.key_levels,
                    'execution_steps': getattr(ps, 'execution_steps', []),
                    'holding_period': ps.holding_period
                }
            )
            standard_signals.append(signal)

            # Track active signal
            self.active_signals.append(ps)

        return standard_signals

    # Helper methods
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using ADX-like calculation"""
        close_prices = df['close'].values
        if len(close_prices) < 20:
            return 0

        # Simple trend strength: slope of linear regression
        x = np.arange(len(close_prices))
        slope, _ = np.polyfit(x, close_prices, 1)

        # Normalize by price
        return slope / np.mean(close_prices) * 100

    def _find_support_levels(self, df: pd.DataFrame, num_levels: int = 3) -> List[float]:
        """Find key support levels"""
        lows = df['low'].values

        # Find local minima
        support_levels = []
        for i in range(10, len(lows) - 10):
            if lows[i] == min(lows[i-10:i+10]):
                support_levels.append(lows[i])

        # Remove duplicates and sort
        support_levels = sorted(list(set(support_levels)))

        return support_levels[-num_levels:] if len(support_levels) >= num_levels else support_levels

    def _find_resistance_levels(self, df: pd.DataFrame, num_levels: int = 3) -> List[float]:
        """Find key resistance levels"""
        highs = df['high'].values

        # Find local maxima
        resistance_levels = []
        for i in range(10, len(highs) - 10):
            if highs[i] == max(highs[i-10:i+10]):
                resistance_levels.append(highs[i])

        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)

        return resistance_levels[:num_levels]

    def _calculate_risk_reward(self, pattern: Dict) -> float:
        """Calculate risk/reward ratio"""
        risk = abs(pattern['entry'] - pattern['stop'])
        reward = abs(pattern['targets'][0] - pattern['entry'])

        return reward / risk if risk > 0 else 0

    def _get_confidence_level(self, confidence: float) -> SignalConfidence:
        """Get confidence level enum"""
        if confidence >= 90:
            return SignalConfidence.ULTRA_HIGH
        elif confidence >= 80:
            return SignalConfidence.HIGH
        elif confidence >= 70:
            return SignalConfidence.MEDIUM
        else:
            return SignalConfidence.LOW

    def _estimate_holding_period(self, pattern: Dict) -> str:
        """Estimate how long to hold the position"""
        pattern_type = pattern['type']

        holding_periods = {
            'triangle_breakout': '2-4 hours',
            'range_breakout': '1-3 hours',
            'double_bottom': '4-8 hours',
            'head_and_shoulders': '6-12 hours',
            'bull_flag': '1-2 hours',
            'bear_flag': '1-2 hours',
            'gartley': '3-6 hours'
        }

        return holding_periods.get(pattern_type, '2-4 hours')

    # Pattern detection helpers (simplified implementations)
    def _is_triangle_pattern(self, df: pd.DataFrame) -> bool:
        """Check if price is forming a triangle"""
        highs = df['high'].rolling(5).max().dropna()
        lows = df['low'].rolling(5).min().dropna()

        if len(highs) < 20:
            return False

        # Check for converging highs and lows
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]

        # Triangle if one is flat and other is sloping, or both converging
        return (abs(high_slope) < 0.1 and low_slope > 0.1) or \
               (high_slope < -0.1 and abs(low_slope) < 0.1) or \
               (high_slope < 0 and low_slope > 0)

    def _is_double_bottom(self, df: pd.DataFrame) -> bool:
        """Check for double bottom pattern"""
        if len(df) < 40:
            return False

        lows = df['low'].values

        # Find two prominent lows
        first_low_idx = np.argmin(lows[:20])
        second_low_idx = np.argmin(lows[20:]) + 20

        first_low = lows[first_low_idx]
        second_low = lows[second_low_idx]

        # Check if lows are similar (within 1%)
        if abs(first_low - second_low) / first_low > 0.01:
            return False

        # Check for peak between lows
        peak_between = max(lows[first_low_idx:second_low_idx])

        return (peak_between - first_low) / first_low > 0.03  # At least 3% peak

    def _determine_regime(self, volatility: float, trend_strength: float) -> str:
        """Determine market regime"""
        if volatility > 0.3:
            return 'volatile'
        elif abs(trend_strength) > 2:
            return 'trending'
        else:
            return 'ranging'

    def _count_higher_highs(self, df: pd.DataFrame) -> int:
        """Count higher highs in recent price action"""
        highs = df['high'].values[-20:]
        count = 0

        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                count += 1

        return count

    def _count_lower_lows(self, df: pd.DataFrame) -> int:
        """Count lower lows in recent price action"""
        lows = df['low'].values[-20:]
        count = 0

        for i in range(1, len(lows)):
            if lows[i] < lows[i-1]:
                count += 1

        return count

    # Placeholder methods for pattern detection
    def _resample_data(self, market_data: Dict, timeframe: str) -> pd.DataFrame:
        """Resample data to different timeframe"""
        # In production, implement proper resampling
        return pd.DataFrame(market_data['candles'])

    def _get_trend(self, data: pd.DataFrame) -> str:
        """Get trend direction"""
        return 'bullish' if self._calculate_trend_strength(data) > 0 else 'bearish'

    def _get_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum"""
        return data['close'].pct_change(10).iloc[-1] * 100

    def _get_nearest_support(self, data: pd.DataFrame) -> float:
        """Get nearest support level"""
        supports = self._find_support_levels(data)
        return supports[-1] if supports else 0

    def _get_nearest_resistance(self, data: pd.DataFrame) -> float:
        """Get nearest resistance level"""
        resistances = self._find_resistance_levels(data)
        return resistances[0] if resistances else 0

    def _calculate_confluence(self, mtf_signals: Dict) -> float:
        """Calculate multi-timeframe confluence score"""
        # Count how many timeframes agree on direction
        bullish_count = sum(1 for tf in mtf_signals.values() if tf['trend'] == 'bullish')
        return bullish_count / len(mtf_signals)

    def _get_primary_trend(self, mtf_signals: Dict) -> str:
        """Get primary trend from higher timeframes"""
        # Weight higher timeframes more
        weights = {'5m': 1, '15m': 2, '1h': 3, '4h': 4, '1d': 5}
        weighted_sum = sum(weights.get(tf, 1) * (1 if data['trend'] == 'bullish' else -1)
                          for tf, data in mtf_signals.items())
        return 'bullish' if weighted_sum > 0 else 'bearish'

    def _identify_key_timeframe(self, mtf_signals: Dict) -> str:
        """Identify most important timeframe"""
        # For now, return 1h as default key timeframe
        return '1h'

    def _check_volume_expansion(self, df: pd.DataFrame) -> bool:
        """Check if volume is expanding"""
        recent_vol = df['volume'].iloc[-5:].mean()
        prev_vol = df['volume'].iloc[-20:-5].mean()
        return recent_vol > prev_vol * 1.5

    def _identify_range(self, df: pd.DataFrame) -> Dict:
        """Identify if price is in a range"""
        highs = df['high'].iloc[-20:]
        lows = df['low'].iloc[-20:]

        resistance = highs.max()
        support = lows.min()

        # Check if price has tested levels multiple times
        resistance_tests = sum(1 for h in highs if h > resistance * 0.99)
        support_tests = sum(1 for l in lows if l < support * 1.01)

        return {
            'is_range': resistance_tests >= 2 and support_tests >= 2,
            'resistance': resistance,
            'support': support,
            'range_width': (resistance - support) / support * 100
        }

    def _is_breaking_range(self, df: pd.DataFrame, range_data: Dict) -> bool:
        """Check if price is breaking out of range"""
        current_price = df['close'].iloc[-1]
        return current_price > range_data['resistance'] * 0.998

    def _find_neckline(self, df: pd.DataFrame, pattern_type: str) -> float:
        """Find neckline for reversal patterns"""
        # Simplified neckline detection
        if pattern_type == 'double_bottom':
            # Find peak between two bottoms
            return df['high'].iloc[-30:-10].max()
        else:  # head and shoulders
            # Find trough levels
            return df['low'].iloc[-30:-10].mean()

    def _is_head_and_shoulders(self, df: pd.DataFrame) -> bool:
        """Detect head and shoulders pattern"""
        # Simplified detection
        if len(df) < 60:
            return False

        highs = df['high'].values[-60:]

        # Find three peaks
        peak1_idx = np.argmax(highs[:20])
        peak2_idx = np.argmax(highs[20:40]) + 20
        peak3_idx = np.argmax(highs[40:]) + 40

        # Check if middle peak is highest (head)
        return highs[peak2_idx] > highs[peak1_idx] and highs[peak2_idx] > highs[peak3_idx]

    def _find_left_shoulder(self, df: pd.DataFrame) -> float:
        """Find left shoulder in H&S pattern"""
        return df['high'].iloc[-60:-40].max()

    def _find_right_shoulder(self, df: pd.DataFrame) -> float:
        """Find right shoulder in H&S pattern"""
        return df['high'].iloc[-20:].max()

    def _is_flag_pattern(self, df: pd.DataFrame) -> bool:
        """Detect flag pattern"""
        # Check for strong move followed by consolidation
        if len(df) < 30:
            return False

        # Check for pole (strong move)
        pole_move = abs(df['close'].iloc[-30] - df['close'].iloc[-20]) / df['close'].iloc[-30]

        # Check for flag (consolidation)
        flag_range = (df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()) / df['close'].iloc[-20]

        return pole_move > 0.03 and flag_range < 0.02

    def _analyze_flag(self, df: pd.DataFrame) -> Dict:
        """Analyze flag pattern details"""
        pole_start = df['close'].iloc[-30]
        pole_end = df['close'].iloc[-20]
        direction = 'bull' if pole_end > pole_start else 'bear'

        flag_high = df['high'].iloc[-20:].max()
        flag_low = df['low'].iloc[-20:].min()

        # Calculate breakout and targets
        pole_height = abs(pole_end - pole_start)

        if direction == 'bull':
            breakout_level = flag_high * 1.001
            targets = [
                breakout_level + pole_height * 0.5,
                breakout_level + pole_height * 1.0,
                breakout_level + pole_height * 1.5
            ]
        else:
            breakout_level = flag_low * 0.999
            targets = [
                breakout_level - pole_height * 0.5,
                breakout_level - pole_height * 1.0,
                breakout_level - pole_height * 1.5
            ]

        return {
            'direction': direction,
            'breakout_level': breakout_level,
            'flag_high': flag_high,
            'flag_low': flag_low,
            'targets': targets,
            'pole_height': pole_height
        }

    def _is_gartley_pattern(self, df: pd.DataFrame) -> bool:
        """Detect Gartley harmonic pattern"""
        # Simplified Gartley detection
        return len(df) > 50 and np.random.random() > 0.9  # Placeholder

    def _calculate_gartley_completion(self, df: pd.DataFrame) -> Dict:
        """Calculate Gartley pattern completion point"""
        # Simplified calculation
        current_price = df['close'].iloc[-1]
        return {
            'X': current_price * 0.95,
            'A': current_price * 0.98,
            'B': current_price * 0.97,
            'C': current_price * 0.99,
            'D': current_price  # Completion point
        }

    def _calculate_triangle_breakout(self, df: pd.DataFrame) -> float:
        """Calculate triangle breakout level"""
        # Find converging trendlines
        highs = df['high'].iloc[-20:]
        lows = df['low'].iloc[-20:]

        # Simple average of recent highs for breakout
        return highs.iloc[-5:].mean()

    def _create_pattern_annotations(self, pattern: Dict) -> List[Dict]:
        """Create pattern-specific annotations"""
        annotations = []

        if pattern['type'] == 'triangle_breakout':
            # Add triangle lines
            annotations.append({
                'type': 'trendline',
                'points': pattern['pattern_data'].get('trendline_points', []),
                'color': 'blue',
                'style': 'solid'
            })

        return annotations

    def _generate_market_narrative(self, context: Dict) -> str:
        """Generate market context narrative"""
        return f"Market is in {context['regime']} regime with {context['trend_direction']} bias. " \
               f"Volatility: {context['volatility']:.1%}, Trend strength: {abs(context['trend_strength']):.1f}"

    def _near_major_level(self, price: float, context: Dict) -> bool:
        """Check if price is near major support/resistance"""
        for level in context['support_levels'] + context['resistance_levels']:
            if abs(price - level) / level < 0.005:  # Within 0.5%
                return True
        return False

    def _remove_conflicts(self, signals: List[ProphetSignal]) -> List[ProphetSignal]:
        """Remove conflicting signals"""
        # Remove signals on same symbol with opposite directions
        filtered = []
        symbols_seen = {}

        for signal in signals:
            if signal.symbol not in symbols_seen:
                filtered.append(signal)
                symbols_seen[signal.symbol] = signal.action
            elif symbols_seen[signal.symbol] == signal.action:
                # Same direction, keep if higher confidence
                if signal.confidence > filtered[-1].confidence:
                    filtered[-1] = signal

        return filtered
