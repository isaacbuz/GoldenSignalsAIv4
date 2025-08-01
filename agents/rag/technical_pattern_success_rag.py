"""
Technical Pattern Success RAG
Historical success rates and optimal parameters for technical patterns
Issue #183: RAG-4: Implement Technical Pattern Success RAG
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of technical patterns"""
    # Chart patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    FLAG_BULL = "flag_bull"
    FLAG_BEAR = "flag_bear"
    CUP_AND_HANDLE = "cup_and_handle"

    # Candlestick patterns
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULL = "engulfing_bullish"
    ENGULFING_BEAR = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"

    # Indicator patterns
    RSI_DIVERGENCE = "rsi_divergence"
    MACD_CROSS = "macd_cross"
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    BOLLINGER_SQUEEZE = "bollinger_squeeze"


class MarketCondition(Enum):
    """Market conditions affecting pattern success"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class PatternInstance:
    """Historical instance of a technical pattern"""
    id: str
    symbol: str
    pattern_type: PatternType
    detected_date: datetime
    pattern_start: datetime
    pattern_end: datetime
    market_condition: MarketCondition
    timeframe: str  # 1m, 5m, 1h, 1d, etc.

    # Pattern characteristics
    pattern_strength: float  # 0-1 score
    volume_confirmation: bool
    pattern_dimensions: Dict[str, float]  # height, width, etc.

    # Entry/exit details
    entry_price: float
    entry_date: datetime
    stop_loss: float
    take_profit_targets: List[float]

    # Outcome
    exit_price: float
    exit_date: datetime
    exit_reason: str  # target_hit, stop_hit, time_exit, pattern_failed
    profit_loss_percent: float
    max_favorable_excursion: float  # Best unrealized profit
    max_adverse_excursion: float  # Worst unrealized loss

    # Context
    market_cap: str  # large, mid, small
    sector: str
    relative_volume: float  # vs average
    news_events: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'pattern_type': self.pattern_type.value,
            'detected_date': self.detected_date.isoformat(),
            'market_condition': self.market_condition.value,
            'timeframe': self.timeframe,
            'pattern_strength': self.pattern_strength,
            'profit_loss_percent': self.profit_loss_percent,
            'exit_reason': self.exit_reason
        }

    def to_embedding(self) -> np.ndarray:
        """Convert pattern to embedding for similarity search"""
        # Create feature vector
        features = [
            # Pattern type (one-hot encoding simplified)
            float(self.pattern_type.value.startswith('triangle')),
            float(self.pattern_type.value.startswith('wedge')),
            float('bull' in self.pattern_type.value),
            float('bear' in self.pattern_type.value),

            # Market condition
            float(self.market_condition == MarketCondition.TRENDING_UP),
            float(self.market_condition == MarketCondition.TRENDING_DOWN),
            float(self.market_condition == MarketCondition.HIGH_VOLATILITY),

            # Pattern characteristics
            self.pattern_strength,
            float(self.volume_confirmation),
            self.relative_volume,

            # Timeframe (encoded)
            1.0 if self.timeframe == '1m' else 0.0,
            0.8 if self.timeframe == '5m' else 0.0,
            0.6 if self.timeframe == '1h' else 0.0,
            0.4 if self.timeframe == '4h' else 0.0,
            0.2 if self.timeframe == '1d' else 0.0,

            # Risk/reward
            (self.take_profit_targets[0] - self.entry_price) / (self.entry_price - self.stop_loss) if self.stop_loss else 2.0
        ]

        return np.array(features)


class PatternSuccessAnalyzer:
    """Analyzes historical success rates of patterns"""

    def __init__(self):
        self.success_thresholds = {
            'high': 0.70,  # 70%+ win rate
            'medium': 0.55,  # 55-70% win rate
            'low': 0.40    # 40-55% win rate
        }

    def calculate_pattern_statistics(self,
                                   patterns: List[PatternInstance]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a pattern type"""
        if not patterns:
            return {'error': 'No patterns to analyze'}

        # Basic statistics
        total_patterns = len(patterns)
        profitable = [p for p in patterns if p.profit_loss_percent > 0]
        win_rate = len(profitable) / total_patterns

        # Profit/loss statistics
        profits = [p.profit_loss_percent for p in patterns]
        avg_profit = np.mean(profits)
        median_profit = np.median(profits)

        # Win/loss analysis
        wins = [p.profit_loss_percent for p in profitable]
        losses = [p.profit_loss_percent for p in patterns if p.profit_loss_percent <= 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Risk/reward
        profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')

        # Pattern reliability by market condition
        condition_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'avg_profit': []})
        for pattern in patterns:
            condition = pattern.market_condition.value
            condition_stats[condition]['count'] += 1
            if pattern.profit_loss_percent > 0:
                condition_stats[condition]['wins'] += 1
            condition_stats[condition]['avg_profit'].append(pattern.profit_loss_percent)

        # Calculate condition success rates
        for condition, stats in condition_stats.items():
            stats['win_rate'] = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
            stats['avg_profit'] = np.mean(stats['avg_profit']) if stats['avg_profit'] else 0

        # Timeframe analysis
        timeframe_stats = defaultdict(lambda: {'count': 0, 'win_rate': 0})
        for pattern in patterns:
            tf = pattern.timeframe
            timeframe_stats[tf]['count'] += 1
            if pattern.profit_loss_percent > 0:
                timeframe_stats[tf]['wins'] = timeframe_stats[tf].get('wins', 0) + 1

        for tf, stats in timeframe_stats.items():
            stats['win_rate'] = stats.get('wins', 0) / stats['count']

        # Exit reason analysis
        exit_reasons = defaultdict(int)
        for pattern in patterns:
            exit_reasons[pattern.exit_reason] += 1

        return {
            'total_patterns': total_patterns,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'median_profit': median_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_profit': max(profits),
            'max_loss': min(profits),
            'condition_performance': dict(condition_stats),
            'timeframe_performance': dict(timeframe_stats),
            'exit_reasons': dict(exit_reasons),
            'reliability_score': self._calculate_reliability_score(
                win_rate, profit_factor, total_patterns
            )
        }

    def _calculate_reliability_score(self, win_rate: float,
                                   profit_factor: float,
                                   sample_size: int) -> float:
        """Calculate overall reliability score for a pattern"""
        # Weight factors
        win_rate_weight = 0.4
        profit_factor_weight = 0.4
        sample_size_weight = 0.2

        # Normalize profit factor (cap at 3)
        normalized_pf = min(profit_factor, 3.0) / 3.0

        # Sample size score (more samples = more reliable)
        sample_score = min(sample_size / 100, 1.0)

        reliability = (
            win_rate * win_rate_weight +
            normalized_pf * profit_factor_weight +
            sample_score * sample_size_weight
        )

        return reliability

    def find_optimal_parameters(self,
                               patterns: List[PatternInstance]) -> Dict[str, Any]:
        """Find optimal parameters for pattern trading"""
        if not patterns:
            return {}

        # Group by pattern strength
        strength_groups = {
            'strong': [p for p in patterns if p.pattern_strength > 0.8],
            'medium': [p for p in patterns if 0.5 < p.pattern_strength <= 0.8],
            'weak': [p for p in patterns if p.pattern_strength <= 0.5]
        }

        optimal_params = {}

        for strength, group in strength_groups.items():
            if group:
                profitable = [p for p in group if p.profit_loss_percent > 0]
                win_rate = len(profitable) / len(group)

                # Calculate optimal stop loss distance
                stop_distances = []
                for p in group:
                    if p.stop_loss and p.entry_price:
                        distance = abs(p.entry_price - p.stop_loss) / p.entry_price
                        stop_distances.append(distance)

                optimal_params[strength] = {
                    'win_rate': win_rate,
                    'avg_profit': np.mean([p.profit_loss_percent for p in group]),
                    'optimal_stop_distance': np.percentile(stop_distances, 25) if stop_distances else 0.02,
                    'sample_size': len(group)
                }

        # Volume confirmation impact
        with_volume = [p for p in patterns if p.volume_confirmation]
        without_volume = [p for p in patterns if not p.volume_confirmation]

        volume_impact = {
            'with_volume_win_rate': len([p for p in with_volume if p.profit_loss_percent > 0]) / len(with_volume) if with_volume else 0,
            'without_volume_win_rate': len([p for p in without_volume if p.profit_loss_percent > 0]) / len(without_volume) if without_volume else 0
        }

        return {
            'strength_analysis': optimal_params,
            'volume_impact': volume_impact,
            'recommended_min_strength': 0.7 if optimal_params.get('strong', {}).get('win_rate', 0) > 0.65 else 0.8
        }


class TechnicalPatternSuccessRAG:
    """
    RAG system for technical pattern success rates and optimization
    Provides historical context for pattern reliability
    """

    def __init__(self, use_mock_db: bool = True):
        """Initialize the Technical Pattern Success RAG"""
        self.use_mock_db = use_mock_db
        self.pattern_analyzer = PatternSuccessAnalyzer()
        self.pattern_instances: List[PatternInstance] = []
        self.pattern_embeddings: Dict[str, np.ndarray] = {}

        if use_mock_db:
            self._load_mock_pattern_history()

    def _load_mock_pattern_history(self):
        """Load mock historical pattern data"""
        # Create diverse pattern examples
        patterns = []

        # Successful bull flag in uptrend
        patterns.append(PatternInstance(
            id="bull_flag_aapl_001",
            symbol="AAPL",
            pattern_type=PatternType.FLAG_BULL,
            detected_date=datetime(2023, 9, 15, 10, 30),
            pattern_start=datetime(2023, 9, 10),
            pattern_end=datetime(2023, 9, 15),
            market_condition=MarketCondition.TRENDING_UP,
            timeframe="1h",
            pattern_strength=0.85,
            volume_confirmation=True,
            pattern_dimensions={'flag_height': 5.0, 'pole_height': 15.0},
            entry_price=175.0,
            entry_date=datetime(2023, 9, 15, 11, 0),
            stop_loss=172.0,
            take_profit_targets=[180.0, 185.0, 190.0],
            exit_price=182.0,
            exit_date=datetime(2023, 9, 18, 14, 0),
            exit_reason="target_hit",
            profit_loss_percent=4.0,
            max_favorable_excursion=5.5,
            max_adverse_excursion=-0.8,
            market_cap="large",
            sector="Technology",
            relative_volume=1.5,
            news_events=["iPhone 15 launch"]
        ))

        # Failed double top in ranging market
        patterns.append(PatternInstance(
            id="double_top_spy_001",
            symbol="SPY",
            pattern_type=PatternType.DOUBLE_TOP,
            detected_date=datetime(2023, 8, 20, 14, 0),
            pattern_start=datetime(2023, 8, 1),
            pattern_end=datetime(2023, 8, 20),
            market_condition=MarketCondition.RANGING,
            timeframe="1d",
            pattern_strength=0.72,
            volume_confirmation=False,
            pattern_dimensions={'peak_height': 450.0, 'neckline': 445.0},
            entry_price=444.0,
            entry_date=datetime(2023, 8, 21, 9, 30),
            stop_loss=446.0,
            take_profit_targets=[440.0, 435.0],
            exit_price=446.5,
            exit_date=datetime(2023, 8, 23, 10, 0),
            exit_reason="stop_hit",
            profit_loss_percent=-0.56,
            max_favorable_excursion=0.9,
            max_adverse_excursion=-0.7,
            market_cap="large",
            sector="Index",
            relative_volume=0.8,
            news_events=["Fed minutes release"]
        ))

        # Successful cup and handle
        patterns.append(PatternInstance(
            id="cup_handle_nvda_001",
            symbol="NVDA",
            pattern_type=PatternType.CUP_AND_HANDLE,
            detected_date=datetime(2023, 7, 10, 10, 0),
            pattern_start=datetime(2023, 5, 15),
            pattern_end=datetime(2023, 7, 10),
            market_condition=MarketCondition.TRENDING_UP,
            timeframe="1d",
            pattern_strength=0.91,
            volume_confirmation=True,
            pattern_dimensions={'cup_depth': 50.0, 'handle_depth': 10.0},
            entry_price=420.0,
            entry_date=datetime(2023, 7, 11, 9, 30),
            stop_loss=410.0,
            take_profit_targets=[450.0, 480.0, 500.0],
            exit_price=485.0,
            exit_date=datetime(2023, 7, 25, 15, 0),
            exit_reason="target_hit",
            profit_loss_percent=15.5,
            max_favorable_excursion=18.0,
            max_adverse_excursion=-1.2,
            market_cap="large",
            sector="Technology",
            relative_volume=2.1,
            news_events=["AI boom", "Earnings beat"]
        ))

        # RSI divergence pattern
        patterns.append(PatternInstance(
            id="rsi_div_tsla_001",
            symbol="TSLA",
            pattern_type=PatternType.RSI_DIVERGENCE,
            detected_date=datetime(2023, 6, 5, 11, 0),
            pattern_start=datetime(2023, 6, 1),
            pattern_end=datetime(2023, 6, 5),
            market_condition=MarketCondition.HIGH_VOLATILITY,
            timeframe="4h",
            pattern_strength=0.78,
            volume_confirmation=True,
            pattern_dimensions={'divergence_strength': 0.8},
            entry_price=185.0,
            entry_date=datetime(2023, 6, 5, 14, 0),
            stop_loss=180.0,
            take_profit_targets=[195.0, 200.0],
            exit_price=198.0,
            exit_date=datetime(2023, 6, 8, 10, 0),
            exit_reason="target_hit",
            profit_loss_percent=7.0,
            max_favorable_excursion=8.5,
            max_adverse_excursion=-1.5,
            market_cap="large",
            sector="Automotive",
            relative_volume=1.8,
            news_events=["Supercharger news"]
        ))

        # Failed wedge pattern
        patterns.append(PatternInstance(
            id="wedge_amd_001",
            symbol="AMD",
            pattern_type=PatternType.WEDGE_RISING,
            detected_date=datetime(2023, 5, 20, 15, 0),
            pattern_start=datetime(2023, 5, 10),
            pattern_end=datetime(2023, 5, 20),
            market_condition=MarketCondition.LOW_VOLATILITY,
            timeframe="1h",
            pattern_strength=0.65,
            volume_confirmation=False,
            pattern_dimensions={'wedge_height': 8.0},
            entry_price=115.0,
            entry_date=datetime(2023, 5, 21, 9, 30),
            stop_loss=117.0,
            take_profit_targets=[110.0, 105.0],
            exit_price=114.0,
            exit_date=datetime(2023, 5, 25, 11, 0),
            exit_reason="time_exit",
            profit_loss_percent=-0.87,
            max_favorable_excursion=2.0,
            max_adverse_excursion=-2.5,
            market_cap="large",
            sector="Technology",
            relative_volume=0.7,
            news_events=[]
        ))

        self.pattern_instances.extend(patterns)
        self._generate_pattern_embeddings()

    def _generate_pattern_embeddings(self):
        """Generate embeddings for all patterns"""
        for pattern in self.pattern_instances:
            self.pattern_embeddings[pattern.id] = pattern.to_embedding()

    def _calculate_similarity(self, embedding1: np.ndarray,
                            embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        if norm_product == 0:
            return 0.0
        return dot_product / norm_product

    async def analyze_pattern(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a detected pattern and provide success probability"""
        # Extract pattern details
        pattern_type = PatternType(pattern_data.get('pattern_type', 'flag_bull'))
        timeframe = pattern_data.get('timeframe', '1h')
        market_condition = MarketCondition(pattern_data.get('market_condition', 'trending_up'))
        pattern_strength = pattern_data.get('pattern_strength', 0.75)
        volume_confirmation = pattern_data.get('volume_confirmation', False)

        # Find similar historical patterns
        similar_patterns = self._find_similar_patterns(
            pattern_type, market_condition, timeframe, pattern_strength
        )

        if not similar_patterns:
            return {
                'pattern_type': pattern_type.value,
                'success_probability': 0.5,
                'confidence': 'low',
                'message': 'No historical data available'
            }

        # Calculate statistics
        stats = self.pattern_analyzer.calculate_pattern_statistics(similar_patterns)

        # Find optimal parameters
        optimal_params = self.pattern_analyzer.find_optimal_parameters(similar_patterns)

        # Generate success probability
        base_probability = stats['win_rate']

        # Adjust for pattern strength
        if pattern_strength > optimal_params.get('recommended_min_strength', 0.7):
            base_probability *= 1.1

        # Adjust for volume confirmation
        if volume_confirmation:
            volume_boost = optimal_params['volume_impact'].get('with_volume_win_rate', 0) - \
                          optimal_params['volume_impact'].get('without_volume_win_rate', 0)
            if volume_boost > 0:
                base_probability *= (1 + volume_boost)

        # Cap probability
        success_probability = min(base_probability, 0.85)

        # Determine confidence level
        sample_size = stats['total_patterns']
        if sample_size >= 50:
            confidence = 'high'
        elif sample_size >= 20:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Get best performing conditions
        best_conditions = self._get_best_conditions(stats['condition_performance'])

        return {
            'pattern_type': pattern_type.value,
            'success_probability': float(success_probability),
            'confidence': confidence,
            'historical_stats': {
                'sample_size': sample_size,
                'avg_win_rate': stats['win_rate'],
                'avg_profit': stats['avg_profit'],
                'profit_factor': stats['profit_factor'],
                'reliability_score': stats['reliability_score']
            },
            'optimal_parameters': {
                'min_pattern_strength': optimal_params.get('recommended_min_strength', 0.7),
                'recommended_stop_distance': optimal_params.get('strength_analysis', {}).get('strong', {}).get('optimal_stop_distance', 0.02),
                'volume_confirmation_important': optimal_params['volume_impact']['with_volume_win_rate'] > optimal_params['volume_impact']['without_volume_win_rate']
            },
            'best_conditions': best_conditions,
            'similar_patterns': [
                {
                    'symbol': p.symbol,
                    'date': p.detected_date.isoformat(),
                    'profit': p.profit_loss_percent,
                    'strength': p.pattern_strength
                }
                for p in similar_patterns[:3]
            ],
            'trading_recommendation': self._generate_trading_recommendation(
                success_probability, stats, optimal_params
            )
        }

    def _find_similar_patterns(self, pattern_type: PatternType,
                              market_condition: MarketCondition,
                              timeframe: str,
                              pattern_strength: float) -> List[PatternInstance]:
        """Find similar historical patterns"""
        # Filter by pattern type first
        same_type = [p for p in self.pattern_instances if p.pattern_type == pattern_type]

        # If not enough same type, include related patterns
        if len(same_type) < 10:
            related_types = self._get_related_pattern_types(pattern_type)
            same_type.extend([p for p in self.pattern_instances
                            if p.pattern_type in related_types])

        # Score similarity
        scored_patterns = []
        for pattern in same_type:
            score = 0

            # Market condition match
            if pattern.market_condition == market_condition:
                score += 0.3

            # Timeframe match
            if pattern.timeframe == timeframe:
                score += 0.2

            # Pattern strength similarity
            strength_diff = abs(pattern.pattern_strength - pattern_strength)
            score += 0.3 * (1 - strength_diff)

            # Add pattern with score
            scored_patterns.append((score, pattern))

        # Sort by similarity score
        scored_patterns.sort(key=lambda x: x[0], reverse=True)

        # Return top similar patterns
        return [p for _, p in scored_patterns[:20]]

    def _get_related_pattern_types(self, pattern_type: PatternType) -> List[PatternType]:
        """Get related pattern types for similarity matching"""
        pattern_families = {
            'reversal': [PatternType.HEAD_AND_SHOULDERS, PatternType.DOUBLE_TOP,
                        PatternType.DOUBLE_BOTTOM],
            'continuation': [PatternType.FLAG_BULL, PatternType.FLAG_BEAR,
                           PatternType.TRIANGLE_ASCENDING, PatternType.TRIANGLE_DESCENDING],
            'wedge': [PatternType.WEDGE_RISING, PatternType.WEDGE_FALLING],
            'indicator': [PatternType.RSI_DIVERGENCE, PatternType.MACD_CROSS,
                         PatternType.GOLDEN_CROSS, PatternType.DEATH_CROSS]
        }

        for family, patterns in pattern_families.items():
            if pattern_type in patterns:
                return [p for p in patterns if p != pattern_type]

        return []

    def _get_best_conditions(self, condition_performance: Dict) -> List[Dict[str, Any]]:
        """Extract best performing conditions"""
        best_conditions = []

        for condition, stats in condition_performance.items():
            if stats['win_rate'] > 0.6 and stats['count'] > 5:
                best_conditions.append({
                    'condition': condition,
                    'win_rate': stats['win_rate'],
                    'avg_profit': stats['avg_profit']
                })

        # Sort by win rate
        best_conditions.sort(key=lambda x: x['win_rate'], reverse=True)
        return best_conditions[:3]

    def _generate_trading_recommendation(self, success_prob: float,
                                       stats: Dict[str, Any],
                                       optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific trading recommendation"""
        recommendation = {
            'action': 'pass',
            'confidence': 'low',
            'position_size': 0,
            'risk_parameters': {},
            'notes': []
        }

        # Determine action based on probability and stats
        if success_prob > 0.65 and stats['profit_factor'] > 1.5:
            recommendation['action'] = 'trade'
            recommendation['confidence'] = 'high'
            recommendation['position_size'] = 0.75
            recommendation['notes'].append("High probability setup with good risk/reward")
        elif success_prob > 0.55 and stats['profit_factor'] > 1.2:
            recommendation['action'] = 'trade'
            recommendation['confidence'] = 'medium'
            recommendation['position_size'] = 0.5
            recommendation['notes'].append("Moderate probability setup")
        else:
            recommendation['notes'].append("Low probability or poor risk/reward")

        # Set risk parameters
        if recommendation['action'] == 'trade':
            stop_distance = optimal_params.get('strength_analysis', {}).get('strong', {}).get('optimal_stop_distance', 0.02)
            recommendation['risk_parameters'] = {
                'stop_loss_percent': stop_distance * 100,
                'position_size_kelly': min(0.25, (success_prob - 0.5) * 2),  # Simplified Kelly
                'max_risk_percent': 1.0,  # Max 1% portfolio risk
                'scale_in_levels': 2 if recommendation['confidence'] == 'medium' else 1
            }

            # Add specific notes
            if stats['avg_win'] > abs(stats['avg_loss']) * 1.5:
                recommendation['notes'].append("Favorable risk/reward ratio")

            if optimal_params['volume_impact']['with_volume_win_rate'] > 0.65:
                recommendation['notes'].append("Volume confirmation critical for success")

        return recommendation


# Demo function
async def demo_pattern_success_rag():
    """Demonstrate the Technical Pattern Success RAG"""
    rag = TechnicalPatternSuccessRAG(use_mock_db=True)

    print("Technical Pattern Success RAG Demo")
    print("="*70)

    # Test 1: Analyze a bull flag pattern
    print("\n1. Analyzing Bull Flag Pattern:")
    print("-"*50)

    bull_flag_data = {
        'pattern_type': 'flag_bull',
        'timeframe': '1h',
        'market_condition': 'trending_up',
        'pattern_strength': 0.82,
        'volume_confirmation': True
    }

    result = await rag.analyze_pattern(bull_flag_data)

    print(f"Pattern: {result['pattern_type']}")
    print(f"Success Probability: {result['success_probability']:.1%}")
    print(f"Confidence: {result['confidence'].upper()}")

    print(f"\nHistorical Statistics:")
    hist = result['historical_stats']
    print(f"  Sample Size: {hist['sample_size']} patterns")
    print(f"  Average Win Rate: {hist['avg_win_rate']:.1%}")
    print(f"  Average Profit: {hist['avg_profit']:.1%}")
    print(f"  Profit Factor: {hist['profit_factor']:.2f}")
    print(f"  Reliability Score: {hist['reliability_score']:.2f}")

    print(f"\nOptimal Parameters:")
    opt = result['optimal_parameters']
    print(f"  Min Pattern Strength: {opt['min_pattern_strength']:.2f}")
    print(f"  Recommended Stop: {opt['recommended_stop_distance']:.1%}")
    print(f"  Volume Important: {'YES' if opt['volume_confirmation_important'] else 'NO'}")

    print(f"\nTrading Recommendation:")
    rec = result['trading_recommendation']
    print(f"  Action: {rec['action'].upper()}")
    print(f"  Confidence: {rec['confidence'].upper()}")
    print(f"  Position Size: {rec['position_size']:.0%}")
    if rec['risk_parameters']:
        print(f"  Stop Loss: {rec['risk_parameters']['stop_loss_percent']:.1f}%")

    # Test 2: Analyze a double top pattern
    print("\n\n2. Analyzing Double Top Pattern:")
    print("-"*50)

    double_top_data = {
        'pattern_type': 'double_top',
        'timeframe': '1d',
        'market_condition': 'ranging',
        'pattern_strength': 0.68,
        'volume_confirmation': False
    }

    result2 = await rag.analyze_pattern(double_top_data)

    print(f"Pattern: {result2['pattern_type']}")
    print(f"Success Probability: {result2['success_probability']:.1%}")
    print(f"Recommendation: {result2['trading_recommendation']['action'].upper()}")
    for note in result2['trading_recommendation']['notes']:
        print(f"  - {note}")

    # Test 3: Pattern comparison
    print("\n\n3. Pattern Performance Summary:")
    print("-"*50)

    patterns_to_test = [
        ('cup_and_handle', 'trending_up'),
        ('rsi_divergence', 'high_volatility'),
        ('wedge_rising', 'low_volatility')
    ]

    for pattern_type, condition in patterns_to_test:
        test_data = {
            'pattern_type': pattern_type,
            'market_condition': condition,
            'pattern_strength': 0.75,
            'volume_confirmation': True,
            'timeframe': '1d'
        }

        result = await rag.analyze_pattern(test_data)
        print(f"\n{pattern_type.upper()} in {condition}:")
        print(f"  Success Rate: {result['success_probability']:.1%}")
        print(f"  Action: {result['trading_recommendation']['action'].upper()}")
        print(f"  Sample Size: {result['historical_stats']['sample_size']}")


if __name__ == "__main__":
    asyncio.run(demo_pattern_success_rag())
