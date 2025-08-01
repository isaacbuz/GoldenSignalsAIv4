"""
Liquidity Prediction Agent
Forecasts market liquidity 1-5 minutes ahead for optimal execution timing
Issue #186: Agent-2: Develop Liquidity Prediction Agent
"""

import asyncio
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Represents a point-in-time order book state"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, int]]  # [(price, size), ...]
    asks: List[Tuple[float, int]]  # [(price, size), ...]

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0.0

    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        if self.bids and self.asks:
            return (self.asks[0][0] + self.bids[0][0]) / 2
        return 0.0

    @property
    def depth_imbalance(self) -> float:
        """Calculate order book imbalance"""
        bid_depth = sum(size for _, size in self.bids[:5])
        ask_depth = sum(size for _, size in self.asks[:5])
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            return (bid_depth - ask_depth) / total_depth
        return 0.0

    @property
    def total_depth(self) -> int:
        """Total size on both sides"""
        return sum(size for _, size in self.bids) + sum(size for _, size in self.asks)


@dataclass
class LiquidityMetrics:
    """Container for liquidity metrics"""
    spread: float
    depth: int
    imbalance: float
    volatility: float
    trade_frequency: float
    average_trade_size: float
    liquidity_score: float  # 0-100, higher is better


class LiquidityPredictionAgent(BaseAgent):
    """
    Agent that predicts market liquidity using order book dynamics
    Uses LSTM-like pattern recognition (simplified for demo)
    """

    def __init__(self, name: str = "LiquidityPredictionAgent",
                 model_path: Optional[str] = None,
                 mcp_market_data = None):
        super().__init__(name=name)
        self.model_path = model_path
        self.mcp = mcp_market_data
        self.prediction_horizons = [1, 2, 3, 5]  # minutes ahead

        # Feature buffers for each symbol
        self.features_buffer: Dict[str, deque] = {}
        self.buffer_size = 20  # Keep 20 time steps

        # Liquidity state tracking
        self.current_liquidity: Dict[str, LiquidityMetrics] = {}
        self.liquidity_history: Dict[str, List[LiquidityMetrics]] = {}

        # Simple model parameters (in production, use trained LSTM)
        self.model_weights = {
            'spread_weight': -0.3,
            'depth_weight': 0.4,
            'imbalance_weight': -0.2,
            'volatility_weight': -0.3,
            'frequency_weight': 0.2,
            'momentum_weight': 0.2
        }

    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent"""
        self.config = config or {}
        logger.info(f"{self.name} initialized")

    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze market data and predict liquidity

        Args:
            market_data: Should contain 'symbol' and 'orderbook' data

        Returns:
            Liquidity predictions and recommendations
        """
        symbol = market_data.get('symbol')
        if not symbol:
            logger.error("No symbol provided in market data")
            return None

        try:
            # Get or create feature buffer for symbol
            if symbol not in self.features_buffer:
                self.features_buffer[symbol] = deque(maxlen=self.buffer_size)
                self.liquidity_history[symbol] = []

            # Extract features from current market data
            features = await self._extract_liquidity_features(market_data)
            self.features_buffer[symbol].append(features)

            # Need enough history for prediction
            if len(self.features_buffer[symbol]) < 5:
                return {
                    'status': 'collecting_data',
                    'symbol': symbol,
                    'samples_collected': len(self.features_buffer[symbol]),
                    'samples_needed': 5
                }

            # Predict liquidity for different horizons
            predictions = await self._predict_liquidity(symbol)

            # Generate execution recommendations
            recommendations = self._generate_execution_recommendations(predictions)

            # Store current liquidity state
            self.current_liquidity[symbol] = self._calculate_current_liquidity(features)
            self.liquidity_history[symbol].append(self.current_liquidity[symbol])

            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_liquidity': self._liquidity_to_dict(self.current_liquidity[symbol]),
                'predictions': predictions,
                'recommendations': recommendations,
                'confidence': self._calculate_prediction_confidence(symbol)
            }

        except Exception as e:
            logger.error(f"Error in liquidity prediction: {e}")
            return None

    async def _extract_liquidity_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract liquidity-relevant features from market data"""
        features = {}

        # Order book features
        if 'orderbook' in market_data:
            ob = market_data['orderbook']
            features['spread'] = ob.get('spread', 0.01)
            features['bid_depth'] = sum(bid.get('size', 0) for bid in ob.get('bids', [])[:5])
            features['ask_depth'] = sum(ask.get('size', 0) for ask in ob.get('asks', [])[:5])
            features['depth_imbalance'] = (features['bid_depth'] - features['ask_depth']) / (
                features['bid_depth'] + features['ask_depth'] + 1)
            features['total_depth'] = features['bid_depth'] + features['ask_depth']
        else:
            # Default features if no order book
            features.update({
                'spread': 0.01,
                'bid_depth': 1000,
                'ask_depth': 1000,
                'depth_imbalance': 0,
                'total_depth': 2000
            })

        # Trade flow features
        features['trade_count'] = market_data.get('trade_count', 10)
        features['avg_trade_size'] = market_data.get('avg_trade_size', 100)
        features['volume'] = market_data.get('volume', 1000000)

        # Price features
        features['price'] = market_data.get('price', 100.0)
        features['volatility'] = market_data.get('volatility', 0.02)
        features['price_momentum'] = market_data.get('price_change', 0.0)

        # Time features
        now = datetime.now()
        features['hour'] = now.hour
        features['minute'] = now.minute
        features['seconds_since_open'] = self._seconds_since_market_open(now)
        features['seconds_to_close'] = self._seconds_to_market_close(now)

        # Market regime features
        features['vix'] = market_data.get('vix', 20.0)
        features['market_sentiment'] = market_data.get('sentiment', 0.0)

        return features

    async def _predict_liquidity(self, symbol: str) -> List[Dict[str, Any]]:
        """Predict liquidity for multiple time horizons"""
        predictions = []

        # Get recent features
        recent_features = list(self.features_buffer[symbol])[-10:]

        for horizon in self.prediction_horizons:
            # Simple prediction model (in production, use LSTM)
            pred = self._simple_liquidity_model(recent_features, horizon)

            predictions.append({
                'minutes_ahead': horizon,
                'liquidity_score': float(pred['score']),
                'expected_spread': float(pred['spread']),
                'expected_depth': int(pred['depth']),
                'confidence': float(pred['confidence']),
                'classification': self._classify_liquidity(pred['score'])
            })

        return predictions

    def _simple_liquidity_model(self, features: List[Dict[str, float]],
                               horizon: int) -> Dict[str, Any]:
        """Simple liquidity prediction model"""
        if not features:
            return {'score': 50, 'spread': 0.01, 'depth': 1000, 'confidence': 0}

        # Calculate trends
        recent = features[-1]

        # Spread trend
        spreads = [f['spread'] for f in features]
        spread_trend = np.polyfit(range(len(spreads)), spreads, 1)[0] if len(spreads) > 1 else 0

        # Depth trend
        depths = [f['total_depth'] for f in features]
        depth_trend = np.polyfit(range(len(depths)), depths, 1)[0] if len(depths) > 1 else 0

        # Volume trend
        volumes = [f['volume'] for f in features]
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0] if len(volumes) > 1 else 0

        # Time decay factor
        time_factor = self._get_time_decay_factor(recent, horizon)

        # Calculate liquidity score (0-100)
        base_score = 50

        # Adjust based on current state
        if recent['spread'] < 0.01:
            base_score += 20
        elif recent['spread'] > 0.05:
            base_score -= 20

        if recent['total_depth'] > 5000:
            base_score += 15
        elif recent['total_depth'] < 1000:
            base_score -= 15

        # Adjust based on trends
        if spread_trend < 0:  # Spread tightening
            base_score += 10 * horizon
        else:
            base_score -= 10 * horizon

        if depth_trend > 0:  # Depth increasing
            base_score += 5 * horizon
        else:
            base_score -= 5 * horizon

        # Apply time decay
        base_score *= time_factor

        # Bound score
        liquidity_score = max(0, min(100, base_score))

        # Predict specific metrics
        predicted_spread = recent['spread'] + (spread_trend * horizon * 60)  # 60 seconds per minute
        predicted_depth = recent['total_depth'] + (depth_trend * horizon * 60)

        # Confidence based on data quality and horizon
        confidence = 0.9 - (0.1 * horizon) - (0.05 * recent['volatility'])
        confidence = max(0.3, min(0.95, confidence))

        return {
            'score': liquidity_score,
            'spread': max(0.001, predicted_spread),
            'depth': max(100, int(predicted_depth)),
            'confidence': confidence
        }

    def _get_time_decay_factor(self, features: Dict[str, float], horizon: int) -> float:
        """Calculate time decay factor for predictions"""
        seconds_to_close = features.get('seconds_to_close', 3600)

        # Liquidity typically decreases near close
        if seconds_to_close < 1800:  # Last 30 minutes
            decay = 0.8 - (0.1 * horizon)
        elif seconds_to_close < 900:  # Last 15 minutes
            decay = 0.6 - (0.15 * horizon)
        else:
            decay = 1.0 - (0.05 * horizon)

        # Also consider time since open
        seconds_since_open = features.get('seconds_since_open', 3600)
        if seconds_since_open < 1800:  # First 30 minutes
            decay *= 0.9  # Slightly lower liquidity at open

        return max(0.3, decay)

    def _classify_liquidity(self, score: float) -> str:
        """Classify liquidity level"""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        elif score >= 20:
            return "poor"
        else:
            return "very_poor"

    def _generate_execution_recommendations(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate specific execution recommendations"""
        # Find best execution window
        best_window = max(predictions, key=lambda p: p['liquidity_score'])

        # Determine execution strategy
        if best_window['liquidity_score'] >= 70:
            strategy = "aggressive"
            slice_size = "large"
            urgency = "can_wait" if best_window['minutes_ahead'] > 1 else "execute_now"
        elif best_window['liquidity_score'] >= 50:
            strategy = "moderate"
            slice_size = "medium"
            urgency = "execute_soon"
        else:
            strategy = "passive"
            slice_size = "small"
            urgency = "wait_for_liquidity"

        # Calculate optimal order sizing
        if best_window['expected_depth'] > 5000:
            max_order_size = best_window['expected_depth'] * 0.1  # 10% of depth
        else:
            max_order_size = best_window['expected_depth'] * 0.05  # 5% of depth

        return {
            'best_execution_window': {
                'minutes_ahead': best_window['minutes_ahead'],
                'liquidity_score': best_window['liquidity_score'],
                'expected_spread': best_window['expected_spread']
            },
            'execution_strategy': strategy,
            'recommended_slice_size': slice_size,
            'max_order_size': int(max_order_size),
            'urgency': urgency,
            'spread_threshold': best_window['expected_spread'] * 1.5,
            'alternative_venues': self._suggest_alternative_venues(best_window)
        }

    def _suggest_alternative_venues(self, prediction: Dict[str, Any]) -> List[str]:
        """Suggest alternative execution venues based on liquidity"""
        venues = []

        if prediction['liquidity_score'] < 50:
            venues.extend(['dark_pool', 'block_trading'])

        if prediction['expected_spread'] > 0.03:
            venues.append('midpoint_matching')

        if prediction['classification'] in ['poor', 'very_poor']:
            venues.append('algorithmic_execution')

        return venues

    def _calculate_current_liquidity(self, features: Dict[str, float]) -> LiquidityMetrics:
        """Calculate current liquidity metrics"""
        # Simple scoring based on features
        spread_score = 100 - min(100, features['spread'] * 2000)  # 0.05 spread = 0 score
        depth_score = min(100, features['total_depth'] / 100)  # 10k depth = 100 score
        imbalance_score = 100 - abs(features['depth_imbalance']) * 100
        volatility_score = 100 - min(100, features['volatility'] * 1000)

        # Weighted average
        liquidity_score = (
            spread_score * 0.3 +
            depth_score * 0.3 +
            imbalance_score * 0.2 +
            volatility_score * 0.2
        )

        return LiquidityMetrics(
            spread=features['spread'],
            depth=int(features['total_depth']),
            imbalance=features['depth_imbalance'],
            volatility=features['volatility'],
            trade_frequency=features['trade_count'],
            average_trade_size=features['avg_trade_size'],
            liquidity_score=liquidity_score
        )

    def _liquidity_to_dict(self, metrics: LiquidityMetrics) -> Dict[str, Any]:
        """Convert liquidity metrics to dictionary"""
        return {
            'spread': metrics.spread,
            'depth': metrics.depth,
            'imbalance': metrics.imbalance,
            'volatility': metrics.volatility,
            'trade_frequency': metrics.trade_frequency,
            'average_trade_size': metrics.average_trade_size,
            'liquidity_score': metrics.liquidity_score,
            'classification': self._classify_liquidity(metrics.liquidity_score)
        }

    def _calculate_prediction_confidence(self, symbol: str) -> float:
        """Calculate overall prediction confidence"""
        if symbol not in self.features_buffer:
            return 0.0

        buffer_fullness = len(self.features_buffer[symbol]) / self.buffer_size

        # Check data quality
        if len(self.liquidity_history.get(symbol, [])) < 5:
            history_factor = 0.5
        else:
            # Check prediction accuracy (simplified)
            recent_scores = [m.liquidity_score for m in self.liquidity_history[symbol][-5:]]
            volatility = np.std(recent_scores) if len(recent_scores) > 1 else 10
            history_factor = max(0.3, 1.0 - volatility / 50)

        return min(0.95, buffer_fullness * history_factor)

    def _seconds_since_market_open(self, now: datetime) -> int:
        """Calculate seconds since market open"""
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now < market_open:
            return 0
        return int((now - market_open).total_seconds())

    def _seconds_to_market_close(self, now: datetime) -> int:
        """Calculate seconds to market close"""
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now > market_close:
            return 0
        return int((market_close - now).total_seconds())


# Demo function
async def demo_liquidity_prediction():
    """Demonstrate the Liquidity Prediction Agent"""
    agent = LiquidityPredictionAgent()
    await agent.initialize()

    print("Liquidity Prediction Agent Demo")
    print("="*60)

    # Simulate different liquidity scenarios
    scenarios = [
        {
            "name": "High Liquidity Period",
            "data": {
                'symbol': 'AAPL',
                'price': 150.00,
                'orderbook': {
                    'bids': [{'price': 149.99, 'size': 1000}, {'price': 149.98, 'size': 1500}],
                    'asks': [{'price': 150.01, 'size': 1000}, {'price': 150.02, 'size': 1500}],
                    'spread': 0.02
                },
                'volume': 50000000,
                'trade_count': 1000,
                'avg_trade_size': 500,
                'volatility': 0.01,
                'vix': 15
            }
        },
        {
            "name": "Low Liquidity Period",
            "data": {
                'symbol': 'AAPL',
                'price': 150.00,
                'orderbook': {
                    'bids': [{'price': 149.95, 'size': 100}, {'price': 149.90, 'size': 200}],
                    'asks': [{'price': 150.05, 'size': 100}, {'price': 150.10, 'size': 200}],
                    'spread': 0.10
                },
                'volume': 5000000,
                'trade_count': 100,
                'avg_trade_size': 50,
                'volatility': 0.03,
                'vix': 25
            }
        },
        {
            "name": "Deteriorating Liquidity",
            "data": {
                'symbol': 'AAPL',
                'price': 150.00,
                'orderbook': {
                    'bids': [{'price': 149.97, 'size': 500}, {'price': 149.95, 'size': 300}],
                    'asks': [{'price': 150.03, 'size': 500}, {'price': 150.05, 'size': 300}],
                    'spread': 0.06
                },
                'volume': 20000000,
                'trade_count': 400,
                'avg_trade_size': 200,
                'volatility': 0.02,
                'vix': 22,
                'price_change': -0.5
            }
        }
    ]

    # Build up some history first
    print("Building liquidity history...")
    for _ in range(5):
        await agent.analyze(scenarios[0]['data'])

    # Now test each scenario
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"Current Spread: ${scenario['data']['orderbook']['spread']:.3f}")
        print(f"Volume: {scenario['data']['volume']:,}")

        result = await agent.analyze(scenario['data'])

        if result and result.get('status') != 'collecting_data':
            print(f"\nCurrent Liquidity:")
            current = result['current_liquidity']
            print(f"  Score: {current['liquidity_score']:.1f}/100 ({current['classification']})")
            print(f"  Depth: {current['depth']:,} shares")
            print(f"  Imbalance: {current['imbalance']:.2%}")

            print(f"\nLiquidity Predictions:")
            for pred in result['predictions']:
                print(f"  +{pred['minutes_ahead']} min: Score {pred['liquidity_score']:.1f} "
                      f"({pred['classification']}) - Spread ${pred['expected_spread']:.3f}")

            print(f"\nExecution Recommendations:")
            rec = result['recommendations']
            print(f"  Best Window: +{rec['best_execution_window']['minutes_ahead']} minutes")
            print(f"  Strategy: {rec['execution_strategy'].upper()}")
            print(f"  Max Order Size: {rec['max_order_size']:,} shares")
            print(f"  Urgency: {rec['urgency'].replace('_', ' ').title()}")

            if rec['alternative_venues']:
                print(f"  Alternative Venues: {', '.join(rec['alternative_venues'])}")

            print(f"\nConfidence: {result['confidence']:.1%}")


if __name__ == "__main__":
    asyncio.run(demo_liquidity_prediction())
