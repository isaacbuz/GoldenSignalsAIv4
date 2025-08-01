"""
Market Regime Classification Agent
Identifies current market regime and adapts strategies accordingly
Issue #185: Agent-1: Develop Market Regime Classification Agent
"""

import asyncio
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_QUIET = "bull_quiet"  # Steady uptrend, low volatility
    BULL_VOLATILE = "bull_volatile"  # Uptrend with high volatility
    BEAR_QUIET = "bear_quiet"  # Steady downtrend, low volatility
    BEAR_VOLATILE = "bear_volatile"  # Downtrend with high volatility
    RANGING_LOW_VOL = "ranging_low_vol"  # Sideways, low volatility
    RANGING_HIGH_VOL = "ranging_high_vol"  # Sideways, high volatility
    TRANSITION = "transition"  # Regime change in progress


class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    EXTREMELY_LOW = "extremely_low"  # <10 VIX
    LOW = "low"  # 10-15 VIX
    NORMAL = "normal"  # 15-20 VIX
    ELEVATED = "elevated"  # 20-30 VIX
    HIGH = "high"  # 30-40 VIX
    EXTREME = "extreme"  # >40 VIX


class TrendStrength(Enum):
    """Trend strength classifications"""
    STRONG_UP = "strong_up"
    MODERATE_UP = "moderate_up"
    WEAK_UP = "weak_up"
    NEUTRAL = "neutral"
    WEAK_DOWN = "weak_down"
    MODERATE_DOWN = "moderate_down"
    STRONG_DOWN = "strong_down"


@dataclass
class RegimeIndicators:
    """Key indicators for regime classification"""
    # Price-based
    price_trend: float  # -1 to 1
    trend_consistency: float  # 0 to 1
    price_momentum: float  # Rate of change

    # Volatility
    current_volatility: float  # Annualized
    volatility_trend: float  # Increasing/decreasing
    volatility_regime: VolatilityRegime

    # Market breadth
    advance_decline_ratio: float
    new_highs_lows_ratio: float
    percent_above_ma: float  # % stocks above 200MA

    # Volume
    volume_trend: float
    volume_volatility_correlation: float

    # Correlations
    sector_correlation: float  # Average correlation between sectors
    asset_correlation: float  # Stock-bond correlation

    # Options
    put_call_ratio: float
    term_structure_slope: float  # VIX9D/VIX

    # Macro
    yield_curve_slope: float
    dollar_strength: float
    commodity_trend: float


class RegimeDetector:
    """Detects market regime based on multiple indicators"""

    def __init__(self):
        self.regime_thresholds = {
            'trend': {
                'strong_up': 0.5,
                'moderate_up': 0.2,
                'neutral': -0.2,
                'moderate_down': -0.5
            },
            'volatility': {
                'low': 15,
                'normal': 20,
                'elevated': 30,
                'high': 40
            }
        }

        # Regime characteristics
        self.regime_profiles = {
            MarketRegime.BULL_QUIET: {
                'trend': (0.3, 1.0),
                'volatility': (0, 20),
                'consistency': (0.7, 1.0),
                'breadth': (0.6, 1.0)
            },
            MarketRegime.BULL_VOLATILE: {
                'trend': (0.2, 1.0),
                'volatility': (20, 100),
                'consistency': (0.3, 0.7),
                'breadth': (0.4, 0.8)
            },
            MarketRegime.BEAR_QUIET: {
                'trend': (-1.0, -0.3),
                'volatility': (0, 25),
                'consistency': (0.7, 1.0),
                'breadth': (0, 0.4)
            },
            MarketRegime.BEAR_VOLATILE: {
                'trend': (-1.0, -0.2),
                'volatility': (25, 100),
                'consistency': (0.3, 0.7),
                'breadth': (0, 0.3)
            },
            MarketRegime.RANGING_LOW_VOL: {
                'trend': (-0.2, 0.2),
                'volatility': (0, 20),
                'consistency': (0, 0.3),
                'breadth': (0.3, 0.7)
            },
            MarketRegime.RANGING_HIGH_VOL: {
                'trend': (-0.3, 0.3),
                'volatility': (20, 100),
                'consistency': (0, 0.4),
                'breadth': (0.2, 0.8)
            }
        }

    def classify_regime(self, indicators: RegimeIndicators) -> Tuple[MarketRegime, float]:
        """Classify market regime based on indicators"""
        scores = {}

        for regime, profile in self.regime_profiles.items():
            score = 0
            weights_sum = 0

            # Trend score
            if profile['trend'][0] <= indicators.price_trend <= profile['trend'][1]:
                score += 0.3
            weights_sum += 0.3

            # Volatility score
            vol = indicators.current_volatility * 100  # Convert to percentage
            if profile['volatility'][0] <= vol <= profile['volatility'][1]:
                score += 0.3
            weights_sum += 0.3

            # Consistency score
            if profile['consistency'][0] <= indicators.trend_consistency <= profile['consistency'][1]:
                score += 0.2
            weights_sum += 0.2

            # Breadth score
            if profile['breadth'][0] <= indicators.percent_above_ma <= profile['breadth'][1]:
                score += 0.2
            weights_sum += 0.2

            scores[regime] = score / weights_sum

        # Get best matching regime
        best_regime = max(scores, key=scores.get)
        confidence = scores[best_regime]

        # Check for transition
        if confidence < 0.6:
            return MarketRegime.TRANSITION, confidence

        return best_regime, confidence

    def detect_regime_change(self, current: MarketRegime,
                           previous: MarketRegime,
                           indicators: RegimeIndicators) -> Dict[str, Any]:
        """Detect and analyze regime changes"""
        if current == previous:
            return {
                'change_detected': False,
                'from_regime': previous,
                'to_regime': current,
                'significance': 'none'
            }

        # Determine significance
        significance = self._assess_change_significance(previous, current)

        # Generate alerts based on change type
        alerts = []
        if significance in ['major', 'critical']:
            alerts.append({
                'type': 'regime_change',
                'severity': significance,
                'message': f"Market regime shift: {previous.value} → {current.value}",
                'action_required': True
            })

        return {
            'change_detected': True,
            'from_regime': previous,
            'to_regime': current,
            'significance': significance,
            'alerts': alerts,
            'recommended_actions': self._get_transition_actions(previous, current)
        }

    def _assess_change_significance(self, from_regime: MarketRegime,
                                  to_regime: MarketRegime) -> str:
        """Assess the significance of regime change"""
        # Critical changes (risk-on to risk-off or vice versa)
        critical_changes = [
            (MarketRegime.BULL_QUIET, MarketRegime.BEAR_VOLATILE),
            (MarketRegime.BULL_VOLATILE, MarketRegime.BEAR_VOLATILE),
            (MarketRegime.BEAR_VOLATILE, MarketRegime.BULL_QUIET)
        ]

        if (from_regime, to_regime) in critical_changes:
            return 'critical'

        # Major changes (significant shift)
        if ('bull' in from_regime.value and 'bear' in to_regime.value) or \
           ('bear' in from_regime.value and 'bull' in to_regime.value):
            return 'major'

        # Moderate changes (volatility shift)
        if from_regime.value.split('_')[0] == to_regime.value.split('_')[0]:
            return 'moderate'

        return 'minor'

    def _get_transition_actions(self, from_regime: MarketRegime,
                              to_regime: MarketRegime) -> List[str]:
        """Get recommended actions for regime transition"""
        actions = []

        # Bull to Bear transition
        if 'bull' in from_regime.value and 'bear' in to_regime.value:
            actions.extend([
                "Reduce long exposure",
                "Increase cash allocation",
                "Consider defensive sectors",
                "Implement stop losses",
                "Review risk limits"
            ])

        # Bear to Bull transition
        elif 'bear' in from_regime.value and 'bull' in to_regime.value:
            actions.extend([
                "Increase long exposure gradually",
                "Focus on quality growth",
                "Reduce defensive positions",
                "Expand risk budget"
            ])

        # Volatility increase
        if 'quiet' in from_regime.value and 'volatile' in to_regime.value:
            actions.extend([
                "Reduce position sizes",
                "Widen stop losses",
                "Consider volatility hedges",
                "Increase monitoring frequency"
            ])

        # Volatility decrease
        elif 'volatile' in from_regime.value and 'quiet' in to_regime.value:
            actions.extend([
                "Increase position sizes",
                "Tighten stop losses",
                "Reduce hedges",
                "Extend holding periods"
            ])

        return actions


class MarketRegimeClassificationAgent:
    """
    Agent that classifies market regime and provides adaptive strategies
    Uses multiple timeframes and indicators for robust classification
    """

    def __init__(self, lookback_periods: Dict[str, int] = None):
        """Initialize the market regime classification agent"""
        self.regime_detector = RegimeDetector()
        self.current_regime = MarketRegime.RANGING_LOW_VOL
        self.regime_history = deque(maxlen=100)
        self.confidence_threshold = 0.6

        # Lookback periods for different calculations
        self.lookback_periods = lookback_periods or {
            'short': 20,  # ~1 month
            'medium': 60,  # ~3 months
            'long': 252   # ~1 year
        }

        # Strategy adjustments per regime
        self.regime_strategies = {
            MarketRegime.BULL_QUIET: {
                'trend_following_weight': 0.7,
                'mean_reversion_weight': 0.1,
                'volatility_trading_weight': 0.0,
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.5,
                'holding_period': 'medium_to_long'
            },
            MarketRegime.BULL_VOLATILE: {
                'trend_following_weight': 0.5,
                'mean_reversion_weight': 0.2,
                'volatility_trading_weight': 0.2,
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 2.0,
                'holding_period': 'short_to_medium'
            },
            MarketRegime.BEAR_QUIET: {
                'trend_following_weight': 0.3,
                'mean_reversion_weight': 0.1,
                'volatility_trading_weight': 0.1,
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 1.2,
                'holding_period': 'short'
            },
            MarketRegime.BEAR_VOLATILE: {
                'trend_following_weight': 0.1,
                'mean_reversion_weight': 0.3,
                'volatility_trading_weight': 0.3,
                'position_size_multiplier': 0.3,
                'stop_loss_multiplier': 2.5,
                'holding_period': 'very_short'
            },
            MarketRegime.RANGING_LOW_VOL: {
                'trend_following_weight': 0.1,
                'mean_reversion_weight': 0.7,
                'volatility_trading_weight': 0.1,
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'holding_period': 'short'
            },
            MarketRegime.RANGING_HIGH_VOL: {
                'trend_following_weight': 0.2,
                'mean_reversion_weight': 0.4,
                'volatility_trading_weight': 0.3,
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.8,
                'holding_period': 'very_short'
            }
        }

    def _calculate_indicators_from_data(self, market_data: Dict[str, Any]) -> RegimeIndicators:
        """Calculate regime indicators from market data"""
        # Extract data
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])
        vix = market_data.get('vix', 20)

        # Price trend (using linear regression slope)
        if len(prices) > 20:
            x = np.arange(len(prices[-self.lookback_periods['short']:]))
            y = prices[-self.lookback_periods['short']:]
            slope = np.polyfit(x, y, 1)[0]
            price_trend = np.tanh(slope * 10)  # Normalize to [-1, 1]
        else:
            price_trend = 0

        # Trend consistency (R-squared of trend)
        if len(prices) > 20:
            y_pred = np.polyval(np.polyfit(x, y, 1), x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            trend_consistency = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            trend_consistency = 0

        # Current volatility (simplified)
        if len(prices) > 2:
            returns = np.diff(prices) / prices[:-1]
            current_volatility = np.std(returns) * np.sqrt(252)
        else:
            current_volatility = 0.2

        # Market breadth (mock data)
        percent_above_ma = market_data.get('percent_above_200ma', 0.5)
        advance_decline = market_data.get('advance_decline_ratio', 1.0)

        # Options data
        put_call_ratio = market_data.get('put_call_ratio', 1.0)

        # Create indicators
        return RegimeIndicators(
            price_trend=price_trend,
            trend_consistency=trend_consistency,
            price_momentum=price_trend * trend_consistency,
            current_volatility=current_volatility,
            volatility_trend=0,  # Simplified
            volatility_regime=self._classify_volatility_regime(vix),
            advance_decline_ratio=advance_decline,
            new_highs_lows_ratio=1.0,  # Mock
            percent_above_ma=percent_above_ma,
            volume_trend=0,  # Simplified
            volume_volatility_correlation=0,
            sector_correlation=0.5,  # Mock
            asset_correlation=0.3,  # Mock
            put_call_ratio=put_call_ratio,
            term_structure_slope=1.0,  # Mock
            yield_curve_slope=1.5,  # Mock
            dollar_strength=0,  # Mock
            commodity_trend=0  # Mock
        )

    def _classify_volatility_regime(self, vix: float) -> VolatilityRegime:
        """Classify volatility regime based on VIX"""
        if vix < 10:
            return VolatilityRegime.EXTREMELY_LOW
        elif vix < 15:
            return VolatilityRegime.LOW
        elif vix < 20:
            return VolatilityRegime.NORMAL
        elif vix < 30:
            return VolatilityRegime.ELEVATED
        elif vix < 40:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    async def classify_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify current market regime"""
        # Calculate indicators
        indicators = self._calculate_indicators_from_data(market_data)

        # Detect regime
        regime, confidence = self.regime_detector.classify_regime(indicators)

        # Check for regime change
        previous_regime = self.current_regime
        regime_change = self.regime_detector.detect_regime_change(
            regime, previous_regime, indicators
        )

        # Update current regime if confidence is high enough
        if confidence >= self.confidence_threshold:
            self.current_regime = regime
            self.regime_history.append({
                'regime': regime,
                'confidence': confidence,
                'timestamp': datetime.now()
            })

        # Get strategy adjustments
        strategy_params = self.regime_strategies.get(
            regime,
            self.regime_strategies[MarketRegime.RANGING_LOW_VOL]
        )

        # Generate insights
        insights = self._generate_regime_insights(regime, indicators)

        return {
            'current_regime': regime.value,
            'confidence': float(confidence),
            'previous_regime': previous_regime.value,
            'regime_change': regime_change,
            'indicators': {
                'price_trend': indicators.price_trend,
                'trend_consistency': indicators.trend_consistency,
                'volatility': indicators.current_volatility,
                'volatility_regime': indicators.volatility_regime.value,
                'market_breadth': indicators.percent_above_ma,
                'put_call_ratio': indicators.put_call_ratio
            },
            'strategy_adjustments': strategy_params,
            'insights': insights,
            'historical_regimes': self._get_regime_statistics(),
            'timestamp': datetime.now().isoformat()
        }

    def _generate_regime_insights(self, regime: MarketRegime,
                                indicators: RegimeIndicators) -> List[str]:
        """Generate insights based on regime and indicators"""
        insights = []

        # Regime-specific insights
        if regime == MarketRegime.BULL_QUIET:
            insights.append("Favorable conditions for trend following")
            insights.append("Consider increasing position sizes")
            insights.append("Focus on momentum stocks")
        elif regime == MarketRegime.BULL_VOLATILE:
            insights.append("Bullish but expect sharp pullbacks")
            insights.append("Use wider stops and smaller positions")
            insights.append("Consider volatility hedges")
        elif regime == MarketRegime.BEAR_QUIET:
            insights.append("Defensive positioning recommended")
            insights.append("Focus on capital preservation")
            insights.append("Look for short opportunities")
        elif regime == MarketRegime.BEAR_VOLATILE:
            insights.append("High risk environment - reduce exposure")
            insights.append("Extreme caution with long positions")
            insights.append("Consider safe havens")
        elif regime == MarketRegime.RANGING_LOW_VOL:
            insights.append("Mean reversion strategies preferred")
            insights.append("Buy support, sell resistance")
            insights.append("Theta strategies may work well")
        elif regime == MarketRegime.RANGING_HIGH_VOL:
            insights.append("Choppy conditions - trade carefully")
            insights.append("Quick profits recommended")
            insights.append("Avoid breakout trades")

        # Indicator-based insights
        if indicators.put_call_ratio > 1.2:
            insights.append("High put/call ratio suggests fear")
        elif indicators.put_call_ratio < 0.7:
            insights.append("Low put/call ratio suggests complacency")

        if indicators.percent_above_ma > 0.8:
            insights.append("Market breadth very strong")
        elif indicators.percent_above_ma < 0.2:
            insights.append("Market breadth very weak")

        return insights

    def _get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics on historical regimes"""
        if not self.regime_history:
            return {}

        # Count regime occurrences
        regime_counts = defaultdict(int)
        for entry in self.regime_history:
            regime_counts[entry['regime'].value] += 1

        # Calculate average duration
        current_regime_start = None
        for i, entry in enumerate(self.regime_history):
            if i == 0 or entry['regime'] != self.regime_history[i-1]['regime']:
                current_regime_start = entry['timestamp']

        current_duration = (datetime.now() - current_regime_start).days if current_regime_start else 0

        return {
            'regime_distribution': dict(regime_counts),
            'total_observations': len(self.regime_history),
            'current_regime_duration_days': current_duration,
            'regime_stability': self._calculate_regime_stability()
        }

    def _calculate_regime_stability(self) -> float:
        """Calculate regime stability (0-1, higher = more stable)"""
        if len(self.regime_history) < 2:
            return 1.0

        # Count regime changes
        changes = 0
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i]['regime'] != self.regime_history[i-1]['regime']:
                changes += 1

        # Normalize by number of observations
        stability = 1 - (changes / len(self.regime_history))
        return max(0, min(1, stability))

    async def get_regime_specific_signals(self, symbol: str,
                                        market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals adapted to current regime"""
        # First classify the regime
        regime_analysis = await self.classify_market_regime(market_data)
        current_regime = MarketRegime(regime_analysis['current_regime'])

        # Get strategy parameters for this regime
        strategy_params = regime_analysis['strategy_adjustments']

        # Generate regime-specific signals
        signals = {
            'symbol': symbol,
            'regime': current_regime.value,
            'signals': []
        }

        # Trend following signal (if applicable)
        if strategy_params['trend_following_weight'] > 0.3:
            signals['signals'].append({
                'type': 'trend_following',
                'weight': strategy_params['trend_following_weight'],
                'action': 'long' if regime_analysis['indicators']['price_trend'] > 0.2 else 'neutral',
                'confidence': abs(regime_analysis['indicators']['price_trend'])
            })

        # Mean reversion signal (if applicable)
        if strategy_params['mean_reversion_weight'] > 0.3:
            signals['signals'].append({
                'type': 'mean_reversion',
                'weight': strategy_params['mean_reversion_weight'],
                'action': 'fade_extremes',
                'confidence': 0.7
            })

        # Volatility trading signal (if applicable)
        if strategy_params['volatility_trading_weight'] > 0.2:
            signals['signals'].append({
                'type': 'volatility',
                'weight': strategy_params['volatility_trading_weight'],
                'action': 'long_volatility' if current_regime.value.endswith('volatile') else 'short_volatility',
                'confidence': 0.6
            })

        # Risk management parameters
        signals['risk_management'] = {
            'position_size': f"{strategy_params['position_size_multiplier']*100:.0f}% of normal",
            'stop_loss': f"{strategy_params['stop_loss_multiplier']*2:.1f}% from entry",
            'holding_period': strategy_params['holding_period']
        }

        return signals


# Demo function
async def demo_regime_classification():
    """Demonstrate the Market Regime Classification Agent"""
    agent = MarketRegimeClassificationAgent()

    print("Market Regime Classification Agent Demo")
    print("="*70)

    # Test different market conditions
    market_scenarios = [
        {
            'name': 'Bull Market - Low Volatility',
            'data': {
                'prices': [100 + i*0.5 + np.random.normal(0, 0.2) for i in range(100)],
                'vix': 12,
                'percent_above_200ma': 0.75,
                'advance_decline_ratio': 2.5,
                'put_call_ratio': 0.7
            }
        },
        {
            'name': 'Bear Market - High Volatility',
            'data': {
                'prices': [100 - i*0.8 + np.random.normal(0, 2) for i in range(100)],
                'vix': 35,
                'percent_above_200ma': 0.20,
                'advance_decline_ratio': 0.4,
                'put_call_ratio': 1.5
            }
        },
        {
            'name': 'Ranging Market - Low Volatility',
            'data': {
                'prices': [100 + 5*np.sin(i/10) + np.random.normal(0, 0.5) for i in range(100)],
                'vix': 15,
                'percent_above_200ma': 0.50,
                'advance_decline_ratio': 1.0,
                'put_call_ratio': 1.0
            }
        }
    ]

    for scenario in market_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print("-"*50)

        result = await agent.classify_market_regime(scenario['data'])

        print(f"Detected Regime: {result['current_regime'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")

        print(f"\nKey Indicators:")
        indicators = result['indicators']
        print(f"  Price Trend: {indicators['price_trend']:+.2f}")
        print(f"  Trend Consistency: {indicators['trend_consistency']:.2f}")
        print(f"  Volatility: {indicators['volatility']:.1%}")
        print(f"  Market Breadth: {indicators['market_breadth']:.1%}")

        print(f"\nStrategy Adjustments:")
        strategy = result['strategy_adjustments']
        print(f"  Trend Following: {strategy['trend_following_weight']:.0%}")
        print(f"  Mean Reversion: {strategy['mean_reversion_weight']:.0%}")
        print(f"  Position Size: {strategy['position_size_multiplier']:.1f}x")

        print(f"\nInsights:")
        for insight in result['insights'][:3]:
            print(f"  • {insight}")

    # Test regime-specific signals
    print("\n\n🎯 Regime-Specific Trading Signals:")
    print("-"*50)

    signals = await agent.get_regime_specific_signals('SPY', market_scenarios[0]['data'])

    print(f"Symbol: {signals['symbol']}")
    print(f"Current Regime: {signals['regime']}")
    print(f"\nSignals:")
    for signal in signals['signals']:
        print(f"  {signal['type'].upper()}:")
        print(f"    Weight: {signal['weight']:.0%}")
        print(f"    Action: {signal['action']}")
        print(f"    Confidence: {signal['confidence']:.1%}")

    print(f"\nRisk Management:")
    rm = signals['risk_management']
    for key, value in rm.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    asyncio.run(demo_regime_classification())
