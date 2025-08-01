"""
Market Regime Classification Agent
Continuously classifies market regime for all other agents
Issue #185: Agent-1: Develop Market Regime Classification Agent
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.base_agent import BaseAgent

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rag.historical_market_context_rag import HistoricalMarketContextRAG

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


class RegimeIndicators:
    """Container for regime indicators"""
    def __init__(self):
        self.vix_level: float = 0.0
        self.vix_change: float = 0.0
        self.breadth: float = 0.0  # Advance/Decline ratio
        self.volume_ratio: float = 1.0
        self.correlation: float = 0.0  # Sector correlation
        self.momentum: float = 0.0  # Price momentum
        self.volatility_percentile: float = 0.0
        self.credit_spread: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'vix_level': self.vix_level,
            'vix_change': self.vix_change,
            'breadth': self.breadth,
            'volume_ratio': self.volume_ratio,
            'correlation': self.correlation,
            'momentum': self.momentum,
            'volatility_percentile': self.volatility_percentile,
            'credit_spread': self.credit_spread
        }


class MarketRegimeClassificationAgent(BaseAgent):
    """
    Agent that continuously classifies market regime (Bull/Bear/Sideways/Crisis)
    Provides regime context to all other agents for better decision making
    """

    def __init__(self, name: str = "MarketRegimeAgent", rag_system: Optional[HistoricalMarketContextRAG] = None):
        super().__init__(name=name)
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.regime_history: List[Dict[str, Any]] = []
        self.indicators = RegimeIndicators()
        self.rag_system = rag_system or HistoricalMarketContextRAG(use_mock_db=True)

        # Adaptive thresholds (will be adjusted based on performance)
        self.thresholds = {
            'vix_crisis': 35.0,
            'vix_high': 25.0,
            'vix_normal': 20.0,
            'breadth_bullish': 2.0,
            'breadth_bearish': 0.5,
            'momentum_strong': 0.02,
            'momentum_weak': -0.02,
            'correlation_high': 0.7,
            'volume_spike': 1.5
        }

        # Performance tracking for adaptation
        self.performance_history: List[Dict[str, Any]] = []
        self.adaptation_counter = 0

    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent with configuration"""
        self.config = config or {}
        if 'thresholds' in self.config:
            self.thresholds.update(self.config['thresholds'])
        logger.info(f"{self.name} initialized with config: {self.config}")

    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze market data and classify regime

        Args:
            market_data: Dictionary containing market indicators

        Returns:
            Analysis results with regime classification
        """
        try:
            # Update indicators from market data
            self._update_indicators(market_data)

            # Get historical context from RAG if available
            historical_context = None
            if self.rag_system:
                historical_context = await self._get_historical_context(market_data)

            # Calculate regime scores
            regime_scores = self._calculate_regime_scores(historical_context)

            # Determine regime with confidence
            self.current_regime, self.regime_confidence = self._determine_regime(regime_scores)

            # Store in history
            regime_record = {
                'timestamp': datetime.now(),
                'regime': self.current_regime.value,
                'confidence': self.regime_confidence,
                'indicators': self.indicators.to_dict(),
                'scores': regime_scores,
                'historical_support': len(historical_context['historical_matches']) if historical_context else 0
            }
            self.regime_history.append(regime_record)

            # Adapt thresholds if enough history
            if len(self.regime_history) > 100 and self.adaptation_counter % 50 == 0:
                await self._adapt_thresholds()
            self.adaptation_counter += 1

            # Generate analysis result
            result = {
                'regime': self.current_regime.value,
                'confidence': self.regime_confidence,
                'indicators': self.indicators.to_dict(),
                'regime_scores': regime_scores,
                'historical_context': historical_context,
                'volatility_forecast': self._forecast_volatility(),
                'risk_level': self._assess_risk_level(),
                'recommended_strategies': self._recommend_strategies(),
                'regime_duration': self._estimate_regime_duration(),
                'transition_probability': self._calculate_transition_probability()
            }

            logger.info(f"Regime classified as {self.current_regime.value} with {self.regime_confidence:.1%} confidence")
            return result

        except Exception as e:
            logger.error(f"Error in regime analysis: {e}")
            return None

    def _update_indicators(self, market_data: Dict[str, Any]):
        """Update regime indicators from market data"""
        self.indicators.vix_level = market_data.get('vix', 20.0)
        self.indicators.vix_change = market_data.get('vix_change', 0.0)
        self.indicators.breadth = market_data.get('advance_decline_ratio', 1.0)
        self.indicators.volume_ratio = market_data.get('volume_ratio', 1.0)
        self.indicators.correlation = market_data.get('sector_correlation', 0.5)
        self.indicators.momentum = market_data.get('spy_momentum', 0.0)
        self.indicators.volatility_percentile = market_data.get('vol_percentile', 50.0)
        self.indicators.credit_spread = market_data.get('credit_spread', 1.0)

    async def _get_historical_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical context from RAG system"""
        current_conditions = {
            'vix': self.indicators.vix_level,
            'spy_change': market_data.get('spy_change', 0),
            'volume_ratio': self.indicators.volume_ratio,
            'events': market_data.get('events', []),
            'regime': self.current_regime.value if self.current_regime != MarketRegime.UNKNOWN else None
        }

        return await self.rag_system.retrieve_similar_scenarios(current_conditions, top_k=5)

    def _calculate_regime_scores(self, historical_context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate probability scores for each regime"""
        scores = {
            MarketRegime.BULL: self._score_bull_regime(historical_context),
            MarketRegime.BEAR: self._score_bear_regime(historical_context),
            MarketRegime.SIDEWAYS: self._score_sideways_regime(historical_context),
            MarketRegime.CRISIS: self._score_crisis_regime(historical_context)
        }

        # Normalize scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}

        return scores

    def _score_bull_regime(self, historical_context: Optional[Dict[str, Any]]) -> float:
        """Calculate bull regime probability"""
        score = 0.0

        # VIX indicators
        if self.indicators.vix_level < self.thresholds['vix_normal']:
            score += 0.2
        if self.indicators.vix_change < -0.05:  # VIX declining
            score += 0.1

        # Market breadth
        if self.indicators.breadth > self.thresholds['breadth_bullish']:
            score += 0.2

        # Momentum
        if self.indicators.momentum > self.thresholds['momentum_strong']:
            score += 0.15

        # Low correlation (healthy market)
        if self.indicators.correlation < self.thresholds['correlation_high']:
            score += 0.1

        # Volume (not too high - orderly advance)
        if self.indicators.volume_ratio < self.thresholds['volume_spike']:
            score += 0.05

        # Historical context
        if historical_context:
            historical_bull = sum(1 for m in historical_context['historical_matches']
                                if m['regime'] == 'bull')
            historical_weight = historical_bull / len(historical_context['historical_matches'])
            score += historical_weight * 0.2

        return score

    def _score_bear_regime(self, historical_context: Optional[Dict[str, Any]]) -> float:
        """Calculate bear regime probability"""
        score = 0.0

        # VIX indicators
        if self.thresholds['vix_normal'] <= self.indicators.vix_level < self.thresholds['vix_crisis']:
            score += 0.15
        if self.indicators.vix_change > 0.05:  # VIX rising
            score += 0.1

        # Market breadth
        if self.indicators.breadth < self.thresholds['breadth_bearish']:
            score += 0.2

        # Momentum
        if self.indicators.momentum < self.thresholds['momentum_weak']:
            score += 0.15

        # High correlation (risk-off)
        if self.indicators.correlation > self.thresholds['correlation_high']:
            score += 0.15

        # Credit spreads widening
        if self.indicators.credit_spread > 1.5:
            score += 0.1

        # Historical context
        if historical_context:
            historical_bear = sum(1 for m in historical_context['historical_matches']
                                if m['regime'] == 'bear')
            historical_weight = historical_bear / len(historical_context['historical_matches'])
            score += historical_weight * 0.15

        return score

    def _score_sideways_regime(self, historical_context: Optional[Dict[str, Any]]) -> float:
        """Calculate sideways regime probability"""
        score = 0.0

        # Moderate VIX
        if self.thresholds['vix_normal'] * 0.8 <= self.indicators.vix_level <= self.thresholds['vix_normal'] * 1.2:
            score += 0.25

        # Neutral breadth
        if 0.8 <= self.indicators.breadth <= 1.2:
            score += 0.25

        # Low momentum
        if abs(self.indicators.momentum) < 0.01:
            score += 0.2

        # Normal volume
        if 0.8 <= self.indicators.volume_ratio <= 1.2:
            score += 0.1

        # Moderate volatility percentile
        if 30 <= self.indicators.volatility_percentile <= 70:
            score += 0.1

        # Historical context
        if historical_context:
            historical_sideways = sum(1 for m in historical_context['historical_matches']
                                    if m['regime'] == 'sideways')
            if historical_context['historical_matches']:
                historical_weight = historical_sideways / len(historical_context['historical_matches'])
                score += historical_weight * 0.1

        return score

    def _score_crisis_regime(self, historical_context: Optional[Dict[str, Any]]) -> float:
        """Calculate crisis regime probability"""
        score = 0.0

        # Extreme VIX
        if self.indicators.vix_level > self.thresholds['vix_crisis']:
            score += 0.3
        if self.indicators.vix_change > 0.2:  # VIX spiking
            score += 0.15

        # Extreme breadth
        if self.indicators.breadth < 0.2:  # Almost everything declining
            score += 0.15

        # Strong negative momentum
        if self.indicators.momentum < -0.05:
            score += 0.1

        # Very high correlation
        if self.indicators.correlation > 0.9:
            score += 0.1

        # Volume spike
        if self.indicators.volume_ratio > 2.0:
            score += 0.1

        # Historical context
        if historical_context:
            historical_crisis = sum(1 for m in historical_context['historical_matches']
                                  if m['regime'] == 'crisis')
            if historical_context['historical_matches']:
                historical_weight = historical_crisis / len(historical_context['historical_matches'])
                score += historical_weight * 0.1

        return score

    def _determine_regime(self, regime_scores: Dict[MarketRegime, float]) -> Tuple[MarketRegime, float]:
        """Determine the most likely regime and confidence"""
        if not regime_scores:
            return MarketRegime.UNKNOWN, 0.0

        # Get regime with highest score
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]

        # Apply some smoothing to avoid rapid regime changes
        if self.regime_history and len(self.regime_history) > 0:
            last_regime = MarketRegime(self.regime_history[-1]['regime'])

            # If regime is changing, require higher confidence
            if best_regime != last_regime and confidence < 0.6:
                # Stay with current regime
                best_regime = last_regime
                confidence = regime_scores.get(last_regime, 0.5)

        return best_regime, confidence

    def _forecast_volatility(self) -> Dict[str, float]:
        """Forecast volatility for different time horizons"""
        base_vol = self.indicators.vix_level

        # Simple forecasts based on regime
        if self.current_regime == MarketRegime.CRISIS:
            return {
                '1_day': base_vol * 1.1,
                '1_week': base_vol * 1.05,
                '1_month': base_vol * 0.9  # Mean reversion
            }
        elif self.current_regime == MarketRegime.BULL:
            return {
                '1_day': base_vol * 0.95,
                '1_week': base_vol * 0.9,
                '1_month': base_vol * 0.85
            }
        else:
            return {
                '1_day': base_vol,
                '1_week': base_vol,
                '1_month': base_vol
            }

    def _assess_risk_level(self) -> str:
        """Assess overall market risk level"""
        if self.current_regime == MarketRegime.CRISIS:
            return "EXTREME"
        elif self.current_regime == MarketRegime.BEAR and self.indicators.vix_level > 30:
            return "HIGH"
        elif self.current_regime == MarketRegime.BEAR:
            return "ELEVATED"
        elif self.current_regime == MarketRegime.SIDEWAYS:
            return "MODERATE"
        else:
            return "LOW"

    def _recommend_strategies(self) -> List[str]:
        """Recommend trading strategies based on regime"""
        strategies = []

        if self.current_regime == MarketRegime.BULL:
            strategies.extend([
                "momentum_long",
                "buy_dips",
                "sell_volatility",
                "growth_stocks"
            ])
        elif self.current_regime == MarketRegime.BEAR:
            strategies.extend([
                "defensive_positions",
                "buy_volatility",
                "short_momentum",
                "quality_stocks"
            ])
        elif self.current_regime == MarketRegime.SIDEWAYS:
            strategies.extend([
                "mean_reversion",
                "range_trading",
                "iron_condors",
                "pairs_trading"
            ])
        elif self.current_regime == MarketRegime.CRISIS:
            strategies.extend([
                "risk_off",
                "tail_hedges",
                "cash_preservation",
                "safe_havens"
            ])

        return strategies

    def _estimate_regime_duration(self) -> Dict[str, Any]:
        """Estimate how long the current regime might last"""
        # Look at historical regime durations
        if not self.regime_history:
            return {"days": "unknown", "confidence": 0.0}

        # Simple estimation based on regime type
        typical_durations = {
            MarketRegime.BULL: 180,  # 6 months
            MarketRegime.BEAR: 90,   # 3 months
            MarketRegime.SIDEWAYS: 60,  # 2 months
            MarketRegime.CRISIS: 30   # 1 month
        }

        estimated_days = typical_durations.get(self.current_regime, 60)

        # Adjust based on how long we've been in current regime
        current_duration = self._get_current_regime_duration()
        remaining = max(0, estimated_days - current_duration)

        return {
            "estimated_remaining_days": remaining,
            "total_expected_days": estimated_days,
            "current_duration_days": current_duration,
            "confidence": 0.6 if self.regime_confidence > 0.7 else 0.3
        }

    def _get_current_regime_duration(self) -> int:
        """Get how many days we've been in the current regime"""
        if not self.regime_history:
            return 0

        days = 0
        for i in range(len(self.regime_history) - 1, -1, -1):
            if self.regime_history[i]['regime'] == self.current_regime.value:
                days += 1
            else:
                break

        return days

    def _calculate_transition_probability(self) -> Dict[str, float]:
        """Calculate probability of transitioning to other regimes"""
        # Simple transition matrix based on current regime
        transition_probs = {
            MarketRegime.BULL: {
                "to_bear": 0.15,
                "to_sideways": 0.25,
                "to_crisis": 0.05,
                "stay": 0.55
            },
            MarketRegime.BEAR: {
                "to_bull": 0.20,
                "to_sideways": 0.30,
                "to_crisis": 0.15,
                "stay": 0.35
            },
            MarketRegime.SIDEWAYS: {
                "to_bull": 0.30,
                "to_bear": 0.25,
                "to_crisis": 0.05,
                "stay": 0.40
            },
            MarketRegime.CRISIS: {
                "to_bull": 0.10,
                "to_bear": 0.50,
                "to_sideways": 0.20,
                "stay": 0.20
            }
        }

        return transition_probs.get(self.current_regime, {})

    async def _adapt_thresholds(self):
        """Adapt thresholds based on performance"""
        # This is a placeholder for adaptive learning
        # In production, this would analyze prediction accuracy
        # and adjust thresholds accordingly
        logger.info("Adapting thresholds based on performance history")

        # Simple adaptation: adjust VIX thresholds based on recent levels
        recent_vix = [h['indicators']['vix_level'] for h in self.regime_history[-50:]]
        if recent_vix:
            avg_vix = np.mean(recent_vix)
            self.thresholds['vix_normal'] = avg_vix
            self.thresholds['vix_high'] = avg_vix * 1.25
            self.thresholds['vix_crisis'] = avg_vix * 1.75

        logger.info(f"Updated thresholds: {self.thresholds}")


# Demo function
async def demo_regime_classification():
    """Demonstrate the Market Regime Classification Agent"""
    # Create agent with RAG system
    rag = HistoricalMarketContextRAG(use_mock_db=True)
    agent = MarketRegimeClassificationAgent(rag_system=rag)

    # Initialize agent
    await agent.initialize()

    # Test different market conditions
    test_scenarios = [
        {
            "name": "Normal Bull Market",
            "data": {
                'vix': 15.5,
                'vix_change': -0.02,
                'advance_decline_ratio': 2.5,
                'volume_ratio': 0.9,
                'sector_correlation': 0.4,
                'spy_momentum': 0.025,
                'spy_change': 0.5,
                'vol_percentile': 30,
                'credit_spread': 0.8,
                'events': ['Positive earnings', 'Economic growth']
            }
        },
        {
            "name": "Bear Market",
            "data": {
                'vix': 28.5,
                'vix_change': 0.15,
                'advance_decline_ratio': 0.4,
                'volume_ratio': 1.8,
                'sector_correlation': 0.85,
                'spy_momentum': -0.03,
                'spy_change': -2.5,
                'vol_percentile': 75,
                'credit_spread': 2.1,
                'events': ['Fed hawkish', 'Recession fears']
            }
        },
        {
            "name": "Market Crisis",
            "data": {
                'vix': 65.0,
                'vix_change': 0.8,
                'advance_decline_ratio': 0.1,
                'volume_ratio': 3.5,
                'sector_correlation': 0.95,
                'spy_momentum': -0.08,
                'spy_change': -7.5,
                'vol_percentile': 99,
                'credit_spread': 5.0,
                'events': ['Black swan event', 'Systemic risk']
            }
        }
    ]

    print("Market Regime Classification Demo")
    print("="*60)

    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-"*40)

        result = await agent.analyze(scenario['data'])

        if result:
            print(f"Regime: {result['regime'].upper()}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"VIX: {scenario['data']['vix']}")
            print(f"Market Breadth: {scenario['data']['advance_decline_ratio']:.2f}")
            print(f"Recommended Strategies: {', '.join(result['recommended_strategies'][:3])}")
            print(f"Volatility Forecast (1 week): {result['volatility_forecast']['1_week']:.1f}")

            if result.get('historical_context'):
                print(f"Historical Support: {len(result['historical_context']['historical_matches'])} similar scenarios")


if __name__ == "__main__":
    asyncio.run(demo_regime_classification())
