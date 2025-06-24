"""
Liquidity Prediction Agent
Predicts market liquidity conditions and optimal execution windows
Issue #186: Agent-2: Develop Liquidity Prediction Agent
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class LiquidityLevel(Enum):
    """Market liquidity levels"""
    VERY_HIGH = "very_high"  # Exceptional liquidity
    HIGH = "high"  # Above average liquidity
    NORMAL = "normal"  # Average liquidity
    LOW = "low"  # Below average liquidity
    VERY_LOW = "very_low"  # Poor liquidity
    DRIED_UP = "dried_up"  # Extreme illiquidity


class MarketSession(Enum):
    """Trading session periods"""
    PRE_MARKET = "pre_market"  # 4:00-9:30 ET
    OPEN_AUCTION = "open_auction"  # 9:30-10:00 ET
    MORNING = "morning"  # 10:00-12:00 ET
    LUNCH = "lunch"  # 12:00-14:00 ET
    AFTERNOON = "afternoon"  # 14:00-15:30 ET
    CLOSE_AUCTION = "close_auction"  # 15:30-16:00 ET
    AFTER_HOURS = "after_hours"  # 16:00-20:00 ET


class OrderType(Enum):
    """Order types for execution"""
    MARKET = "market"
    LIMIT = "limit"
    ICEBERG = "iceberg"
    VWAP = "vwap"
    TWAP = "twap"
    MOC = "moc"  # Market on close
    LOC = "loc"  # Limit on close


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity metrics"""
    # Volume metrics
    current_volume: float
    average_volume: float
    volume_ratio: float  # Current/Average
    volume_profile: Dict[str, float]  # Volume by price level
    
    # Spread metrics
    bid_ask_spread: float
    spread_volatility: float
    average_spread: float
    
    # Depth metrics
    bid_depth: Dict[float, float]  # Price -> Size
    ask_depth: Dict[float, float]  # Price -> Size
    total_bid_depth: float
    total_ask_depth: float
    depth_imbalance: float  # (Bid-Ask)/(Bid+Ask)
    
    # Market impact
    estimated_impact_bps: float  # Basis points per $1M traded
    kyle_lambda: float  # Price impact coefficient
    resilience_factor: float  # Speed of recovery
    
    # Microstructure
    trade_size_distribution: Dict[str, float]
    order_arrival_rate: float  # Orders per minute
    cancel_rate: float  # Cancellation rate
    
    # Time-based
    session: MarketSession
    minutes_to_close: int
    is_news_time: bool
    is_economic_release: bool


@dataclass
class ExecutionRecommendation:
    """Execution strategy recommendation"""
    strategy: OrderType
    urgency: str  # immediate, patient, opportunistic
    slice_size: float  # Percentage of total order
    time_horizon: int  # Minutes
    price_limit: Optional[float]
    dark_pool_percentage: float
    expected_cost_bps: float
    confidence: float
    notes: List[str]


class LiquidityAnalyzer:
    """Analyzes market liquidity conditions"""
    
    def __init__(self):
        # Liquidity thresholds by market cap
        self.liquidity_thresholds = {
            'large_cap': {
                'volume_ratio': {'very_high': 1.5, 'high': 1.2, 'normal': 0.8, 'low': 0.5},
                'spread_bps': {'very_high': 5, 'high': 10, 'normal': 20, 'low': 50},
                'depth_ratio': {'very_high': 2.0, 'high': 1.5, 'normal': 1.0, 'low': 0.5}
            },
            'mid_cap': {
                'volume_ratio': {'very_high': 2.0, 'high': 1.5, 'normal': 0.7, 'low': 0.3},
                'spread_bps': {'very_high': 10, 'high': 25, 'normal': 50, 'low': 100},
                'depth_ratio': {'very_high': 1.5, 'high': 1.2, 'normal': 0.8, 'low': 0.4}
            },
            'small_cap': {
                'volume_ratio': {'very_high': 3.0, 'high': 2.0, 'normal': 0.5, 'low': 0.2},
                'spread_bps': {'very_high': 25, 'high': 50, 'normal': 100, 'low': 200},
                'depth_ratio': {'very_high': 1.0, 'high': 0.8, 'normal': 0.5, 'low': 0.2}
            }
        }
        
        # Optimal execution windows
        self.optimal_windows = {
            MarketSession.OPEN_AUCTION: {'start': time(9, 30), 'end': time(10, 0), 'liquidity_mult': 2.5},
            MarketSession.MORNING: {'start': time(10, 0), 'end': time(11, 30), 'liquidity_mult': 1.5},
            MarketSession.AFTERNOON: {'start': time(14, 30), 'end': time(15, 30), 'liquidity_mult': 1.3},
            MarketSession.CLOSE_AUCTION: {'start': time(15, 30), 'end': time(16, 0), 'liquidity_mult': 2.0}
        }
    
    def classify_liquidity_level(self, metrics: LiquidityMetrics, 
                                market_cap: str = 'large_cap') -> Tuple[LiquidityLevel, float]:
        """Classify current liquidity level"""
        thresholds = self.liquidity_thresholds[market_cap]
        
        # Score based on multiple factors
        scores = {
            'volume': self._score_volume(metrics.volume_ratio, thresholds['volume_ratio']),
            'spread': self._score_spread(metrics.bid_ask_spread, metrics.average_spread, thresholds['spread_bps']),
            'depth': self._score_depth(metrics.total_bid_depth, metrics.total_ask_depth, thresholds['depth_ratio'])
        }
        
        # Weight the scores
        weights = {'volume': 0.4, 'spread': 0.3, 'depth': 0.3}
        total_score = sum(scores[k] * weights[k] for k in scores)
        
        # Classify based on score
        if total_score >= 0.8:
            level = LiquidityLevel.VERY_HIGH
        elif total_score >= 0.6:
            level = LiquidityLevel.HIGH
        elif total_score >= 0.4:
            level = LiquidityLevel.NORMAL
        elif total_score >= 0.2:
            level = LiquidityLevel.LOW
        elif total_score >= 0.1:
            level = LiquidityLevel.VERY_LOW
        else:
            level = LiquidityLevel.DRIED_UP
        
        return level, total_score
    
    def _score_volume(self, volume_ratio: float, thresholds: Dict[str, float]) -> float:
        """Score volume liquidity (0-1)"""
        if volume_ratio >= thresholds['very_high']:
            return 1.0
        elif volume_ratio >= thresholds['high']:
            return 0.8
        elif volume_ratio >= thresholds['normal']:
            return 0.6
        elif volume_ratio >= thresholds['low']:
            return 0.3
        else:
            return 0.1
    
    def _score_spread(self, current_spread: float, avg_spread: float, 
                     thresholds: Dict[str, float]) -> float:
        """Score spread liquidity (0-1, lower spread = better)"""
        spread_bps = current_spread * 10000  # Convert to basis points
        
        if spread_bps <= thresholds['very_high']:
            return 1.0
        elif spread_bps <= thresholds['high']:
            return 0.8
        elif spread_bps <= thresholds['normal']:
            return 0.6
        elif spread_bps <= thresholds['low']:
            return 0.3
        else:
            return 0.1
    
    def _score_depth(self, bid_depth: float, ask_depth: float, 
                    thresholds: Dict[str, float]) -> float:
        """Score market depth liquidity (0-1)"""
        total_depth = bid_depth + ask_depth
        depth_ratio = total_depth / 1000000  # Normalize to millions
        
        if depth_ratio >= thresholds['very_high']:
            return 1.0
        elif depth_ratio >= thresholds['high']:
            return 0.8
        elif depth_ratio >= thresholds['normal']:
            return 0.6
        elif depth_ratio >= thresholds['low']:
            return 0.3
        else:
            return 0.1
    
    def estimate_market_impact(self, order_size: float, 
                             metrics: LiquidityMetrics,
                             urgency: str = 'normal') -> Dict[str, float]:
        """Estimate market impact for order execution"""
        # Base impact using square-root model
        adv_percentage = order_size / metrics.average_volume
        base_impact_bps = 10 * np.sqrt(adv_percentage * 100)
        
        # Adjust for current liquidity
        liquidity_mult = metrics.volume_ratio
        if liquidity_mult < 0.5:
            liquidity_mult = 2.0  # Double impact in low liquidity
        elif liquidity_mult > 1.5:
            liquidity_mult = 0.7  # Reduced impact in high liquidity
        else:
            liquidity_mult = 1.0
        
        # Adjust for urgency
        urgency_mult = {
            'immediate': 1.5,
            'normal': 1.0,
            'patient': 0.7,
            'opportunistic': 0.5
        }.get(urgency, 1.0)
        
        # Adjust for spread
        spread_mult = max(1.0, metrics.bid_ask_spread / metrics.average_spread)
        
        # Calculate total impact
        total_impact_bps = base_impact_bps * liquidity_mult * urgency_mult * spread_mult
        
        # Estimate components
        permanent_impact = total_impact_bps * 0.6  # Permanent price move
        temporary_impact = total_impact_bps * 0.4  # Temporary impact
        
        return {
            'total_impact_bps': total_impact_bps,
            'permanent_impact_bps': permanent_impact,
            'temporary_impact_bps': temporary_impact,
            'spread_cost_bps': metrics.bid_ask_spread * 5000,  # Half spread in bps
            'total_cost_bps': total_impact_bps + metrics.bid_ask_spread * 5000
        }
    
    def recommend_execution_strategy(self, order_size: float,
                                   metrics: LiquidityMetrics,
                                   urgency: str = 'normal',
                                   risk_tolerance: str = 'medium') -> ExecutionRecommendation:
        """Recommend optimal execution strategy"""
        # Classify liquidity
        liquidity_level, score = self.classify_liquidity_level(metrics)
        
        # Calculate order size relative to ADV
        adv_percentage = order_size / metrics.average_volume
        
        # Default recommendation
        recommendation = ExecutionRecommendation(
            strategy=OrderType.LIMIT,
            urgency=urgency,
            slice_size=0.1,
            time_horizon=60,
            price_limit=None,
            dark_pool_percentage=0.0,
            expected_cost_bps=50,
            confidence=0.5,
            notes=[]
        )
        
        # Large order (>5% ADV) strategies
        if adv_percentage > 0.05:
            if liquidity_level in [LiquidityLevel.VERY_HIGH, LiquidityLevel.HIGH]:
                recommendation.strategy = OrderType.VWAP
                recommendation.slice_size = 0.05
                recommendation.time_horizon = 240  # 4 hours
                recommendation.dark_pool_percentage = 0.3
                recommendation.notes.append("Large order: Use VWAP with dark pool")
            else:
                recommendation.strategy = OrderType.ICEBERG
                recommendation.slice_size = 0.02
                recommendation.time_horizon = 480  # Full day
                recommendation.dark_pool_percentage = 0.5
                recommendation.notes.append("Large order in thin market: Use iceberg")
        
        # Medium order (1-5% ADV)
        elif adv_percentage > 0.01:
            if liquidity_level == LiquidityLevel.VERY_HIGH:
                recommendation.strategy = OrderType.LIMIT
                recommendation.slice_size = 0.25
                recommendation.time_horizon = 30
                recommendation.notes.append("Good liquidity: Aggressive limit orders")
            else:
                recommendation.strategy = OrderType.TWAP
                recommendation.slice_size = 0.1
                recommendation.time_horizon = 120
                recommendation.notes.append("Medium order: TWAP execution")
        
        # Small order (<1% ADV)
        else:
            if liquidity_level in [LiquidityLevel.VERY_HIGH, LiquidityLevel.HIGH]:
                recommendation.strategy = OrderType.MARKET
                recommendation.slice_size = 1.0
                recommendation.time_horizon = 1
                recommendation.notes.append("Small order, good liquidity: Market order")
            else:
                recommendation.strategy = OrderType.LIMIT
                recommendation.slice_size = 0.5
                recommendation.time_horizon = 15
                recommendation.notes.append("Small order: Patient limit order")
        
        # Adjust for urgency
        if urgency == 'immediate':
            recommendation.slice_size = min(1.0, recommendation.slice_size * 2)
            recommendation.time_horizon = max(1, recommendation.time_horizon // 2)
            recommendation.notes.append("Urgent: Accelerated execution")
        elif urgency == 'patient':
            recommendation.slice_size = recommendation.slice_size * 0.5
            recommendation.time_horizon = recommendation.time_horizon * 2
            recommendation.notes.append("Patient: Extended execution window")
        
        # Estimate costs
        impact_est = self.estimate_market_impact(order_size, metrics, urgency)
        recommendation.expected_cost_bps = impact_est['total_cost_bps']
        
        # Set confidence based on liquidity score
        recommendation.confidence = score
        
        return recommendation


class LiquidityPredictionAgent:
    """
    Agent that predicts liquidity conditions and recommends execution strategies
    Helps minimize market impact and execution costs
    """
    
    def __init__(self):
        """Initialize the liquidity prediction agent"""
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.historical_patterns = {}
        self.intraday_patterns = self._load_intraday_patterns()
        self.event_calendar = {}
        
        # Track recent predictions
        self.prediction_history = deque(maxlen=100)
        self.accuracy_tracker = {'correct': 0, 'total': 0}
    
    def _load_intraday_patterns(self) -> Dict[MarketSession, Dict[str, float]]:
        """Load typical intraday liquidity patterns"""
        return {
            MarketSession.PRE_MARKET: {
                'typical_volume_pct': 0.02,
                'typical_spread_mult': 3.0,
                'volatility': 'high'
            },
            MarketSession.OPEN_AUCTION: {
                'typical_volume_pct': 0.15,
                'typical_spread_mult': 1.5,
                'volatility': 'very_high'
            },
            MarketSession.MORNING: {
                'typical_volume_pct': 0.35,
                'typical_spread_mult': 1.0,
                'volatility': 'medium'
            },
            MarketSession.LUNCH: {
                'typical_volume_pct': 0.15,
                'typical_spread_mult': 1.2,
                'volatility': 'low'
            },
            MarketSession.AFTERNOON: {
                'typical_volume_pct': 0.25,
                'typical_spread_mult': 1.1,
                'volatility': 'medium'
            },
            MarketSession.CLOSE_AUCTION: {
                'typical_volume_pct': 0.08,
                'typical_spread_mult': 1.3,
                'volatility': 'high'
            }
        }
    
    def _get_current_session(self, current_time: datetime) -> MarketSession:
        """Determine current market session"""
        market_time = current_time.time()
        
        if market_time < time(9, 30):
            return MarketSession.PRE_MARKET
        elif market_time < time(10, 0):
            return MarketSession.OPEN_AUCTION
        elif market_time < time(12, 0):
            return MarketSession.MORNING
        elif market_time < time(14, 0):
            return MarketSession.LUNCH
        elif market_time < time(15, 30):
            return MarketSession.AFTERNOON
        elif market_time < time(16, 0):
            return MarketSession.CLOSE_AUCTION
        else:
            return MarketSession.AFTER_HOURS
    
    def _calculate_liquidity_metrics(self, market_data: Dict[str, Any],
                                   current_time: datetime) -> LiquidityMetrics:
        """Calculate current liquidity metrics from market data"""
        # Extract data
        current_volume = market_data.get('volume', 1000000)
        avg_volume = market_data.get('avg_volume', 10000000)
        bid = market_data.get('bid', 100.0)
        ask = market_data.get('ask', 100.1)
        
        # Calculate spread
        bid_ask_spread = (ask - bid) / ((ask + bid) / 2)
        
        # Mock order book depth
        bid_depth = {
            bid: 10000,
            bid - 0.01: 15000,
            bid - 0.02: 20000,
            bid - 0.03: 25000,
            bid - 0.04: 30000
        }
        
        ask_depth = {
            ask: 12000,
            ask + 0.01: 18000,
            ask + 0.02: 22000,
            ask + 0.03: 28000,
            ask + 0.04: 35000
        }
        
        total_bid_depth = sum(bid_depth.values())
        total_ask_depth = sum(ask_depth.values())
        
        # Get current session
        session = self._get_current_session(current_time)
        
        # Minutes to close
        close_time = datetime.combine(current_time.date(), time(16, 0))
        minutes_to_close = max(0, (close_time - current_time).seconds // 60)
        
        return LiquidityMetrics(
            current_volume=current_volume,
            average_volume=avg_volume,
            volume_ratio=current_volume / avg_volume if avg_volume > 0 else 0,
            volume_profile={},  # Simplified
            bid_ask_spread=bid_ask_spread,
            spread_volatility=0.0002,  # Mock
            average_spread=0.0001,  # Mock
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            total_bid_depth=total_bid_depth,
            total_ask_depth=total_ask_depth,
            depth_imbalance=(total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth),
            estimated_impact_bps=10.0,  # Mock
            kyle_lambda=0.1,  # Mock
            resilience_factor=0.8,  # Mock
            trade_size_distribution={'small': 0.6, 'medium': 0.3, 'large': 0.1},
            order_arrival_rate=50.0,  # Orders per minute
            cancel_rate=0.3,  # 30% cancellation
            session=session,
            minutes_to_close=minutes_to_close,
            is_news_time=False,  # Mock
            is_economic_release=False  # Mock
        )
    
    async def predict_liquidity(self, symbol: str, 
                              market_data: Dict[str, Any],
                              forecast_horizon: int = 30) -> Dict[str, Any]:
        """Predict future liquidity conditions"""
        current_time = datetime.now()
        current_metrics = self._calculate_liquidity_metrics(market_data, current_time)
        
        # Classify current liquidity
        current_level, current_score = self.liquidity_analyzer.classify_liquidity_level(current_metrics)
        
        # Get session patterns
        session_pattern = self.intraday_patterns.get(current_metrics.session, {})
        
        # Predict future liquidity
        future_time = current_time + timedelta(minutes=forecast_horizon)
        future_session = self._get_current_session(future_time)
        future_pattern = self.intraday_patterns.get(future_session, {})
        
        # Estimate future metrics
        future_volume_mult = future_pattern.get('typical_volume_pct', 0.1) / \
                           session_pattern.get('typical_volume_pct', 0.1)
        future_spread_mult = future_pattern.get('typical_spread_mult', 1.0) / \
                           session_pattern.get('typical_spread_mult', 1.0)
        
        # Predict future liquidity level
        predicted_volume_ratio = current_metrics.volume_ratio * future_volume_mult
        predicted_spread = current_metrics.bid_ask_spread * future_spread_mult
        
        # Determine predicted level
        if predicted_volume_ratio > 1.2 and predicted_spread < 0.0002:
            predicted_level = LiquidityLevel.HIGH
            confidence = 0.8
        elif predicted_volume_ratio < 0.5 or predicted_spread > 0.0005:
            predicted_level = LiquidityLevel.LOW
            confidence = 0.7
        else:
            predicted_level = LiquidityLevel.NORMAL
            confidence = 0.85
        
        # Identify optimal execution windows
        optimal_windows = self._identify_optimal_windows(current_time)
        
        # Generate prediction
        prediction = {
            'symbol': symbol,
            'current_liquidity': {
                'level': current_level.value,
                'score': current_score,
                'volume_ratio': current_metrics.volume_ratio,
                'spread_bps': current_metrics.bid_ask_spread * 10000,
                'session': current_metrics.session.value
            },
            'predicted_liquidity': {
                'level': predicted_level.value,
                'confidence': confidence,
                'forecast_horizon_min': forecast_horizon,
                'expected_volume_ratio': predicted_volume_ratio,
                'expected_spread_bps': predicted_spread * 10000
            },
            'optimal_execution_windows': optimal_windows,
            'market_microstructure': {
                'depth_imbalance': current_metrics.depth_imbalance,
                'order_arrival_rate': current_metrics.order_arrival_rate,
                'cancel_rate': current_metrics.cancel_rate
            },
            'warnings': self._generate_liquidity_warnings(current_metrics, predicted_level),
            'timestamp': current_time.isoformat()
        }
        
        # Track prediction
        self.prediction_history.append(prediction)
        
        return prediction
    
    def _identify_optimal_windows(self, current_time: datetime) -> List[Dict[str, Any]]:
        """Identify optimal execution windows for the day"""
        windows = []
        current_date = current_time.date()
        
        for session, params in self.liquidity_analyzer.optimal_windows.items():
            window_start = datetime.combine(current_date, params['start'])
            window_end = datetime.combine(current_date, params['end'])
            
            # Only include future windows
            if window_end > current_time:
                windows.append({
                    'session': session.value,
                    'start': window_start.isoformat(),
                    'end': window_end.isoformat(),
                    'liquidity_multiplier': params['liquidity_mult'],
                    'recommended_for': self._get_window_recommendation(session)
                })
        
        return sorted(windows, key=lambda x: x['start'])
    
    def _get_window_recommendation(self, session: MarketSession) -> str:
        """Get execution recommendation for session"""
        recommendations = {
            MarketSession.OPEN_AUCTION: "Large orders, price discovery",
            MarketSession.MORNING: "General execution, good liquidity",
            MarketSession.LUNCH: "Avoid if possible, low liquidity",
            MarketSession.AFTERNOON: "Closing positions, moderate liquidity",
            MarketSession.CLOSE_AUCTION: "MOC/LOC orders, benchmarking"
        }
        return recommendations.get(session, "Monitor liquidity carefully")
    
    def _generate_liquidity_warnings(self, metrics: LiquidityMetrics,
                                   predicted_level: LiquidityLevel) -> List[str]:
        """Generate liquidity warnings"""
        warnings = []
        
        # Volume warnings
        if metrics.volume_ratio < 0.3:
            warnings.append("Very low volume - expect high impact")
        elif metrics.volume_ratio < 0.5:
            warnings.append("Below average volume - trade carefully")
        
        # Spread warnings
        if metrics.bid_ask_spread > 0.001:  # 10 bps
            warnings.append("Wide spreads - consider limit orders")
        
        # Depth warnings
        if abs(metrics.depth_imbalance) > 0.3:
            if metrics.depth_imbalance > 0:
                warnings.append("Bid-heavy book - potential support")
            else:
                warnings.append("Ask-heavy book - potential resistance")
        
        # Time warnings
        if metrics.minutes_to_close < 30:
            warnings.append("Near market close - expect volatility")
        
        # Future warnings
        if predicted_level in [LiquidityLevel.LOW, LiquidityLevel.VERY_LOW]:
            warnings.append("Deteriorating liquidity expected")
        
        return warnings
    
    async def recommend_execution(self, symbol: str,
                                order_size: float,
                                side: str,  # 'buy' or 'sell'
                                market_data: Dict[str, Any],
                                urgency: str = 'normal',
                                constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Recommend optimal execution strategy"""
        current_time = datetime.now()
        
        # Get current liquidity metrics
        metrics = self._calculate_liquidity_metrics(market_data, current_time)
        
        # Get liquidity prediction
        liquidity_forecast = await self.predict_liquidity(symbol, market_data)
        
        # Get execution recommendation
        exec_rec = self.liquidity_analyzer.recommend_execution_strategy(
            order_size, metrics, urgency
        )
        
        # Calculate execution schedule
        schedule = self._create_execution_schedule(
            order_size, exec_rec, metrics, constraints
        )
        
        # Risk assessment
        risk_assessment = self._assess_execution_risk(
            order_size, metrics, exec_rec
        )
        
        return {
            'symbol': symbol,
            'order_size': order_size,
            'side': side,
            'liquidity_assessment': liquidity_forecast,
            'execution_strategy': {
                'recommended_type': exec_rec.strategy.value,
                'urgency': exec_rec.urgency,
                'expected_cost_bps': exec_rec.expected_cost_bps,
                'confidence': exec_rec.confidence,
                'notes': exec_rec.notes
            },
            'execution_schedule': schedule,
            'risk_assessment': risk_assessment,
            'alternative_strategies': self._get_alternative_strategies(
                order_size, metrics, urgency
            ),
            'timestamp': current_time.isoformat()
        }
    
    def _create_execution_schedule(self, order_size: float,
                                 recommendation: ExecutionRecommendation,
                                 metrics: LiquidityMetrics,
                                 constraints: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create detailed execution schedule"""
        schedule = []
        remaining_size = order_size
        current_time = datetime.now()
        
        # Calculate slice sizes
        slice_size = order_size * recommendation.slice_size
        num_slices = int(np.ceil(order_size / slice_size))
        time_between_slices = recommendation.time_horizon / num_slices
        
        for i in range(num_slices):
            slice_time = current_time + timedelta(minutes=i * time_between_slices)
            size = min(slice_size, remaining_size)
            
            schedule.append({
                'slice_number': i + 1,
                'time': slice_time.isoformat(),
                'size': size,
                'percentage': (size / order_size) * 100,
                'order_type': recommendation.strategy.value,
                'venue': 'dark_pool' if i % 3 == 0 and recommendation.dark_pool_percentage > 0 else 'lit_market'
            })
            
            remaining_size -= size
            if remaining_size <= 0:
                break
        
        return schedule
    
    def _assess_execution_risk(self, order_size: float,
                             metrics: LiquidityMetrics,
                             recommendation: ExecutionRecommendation) -> Dict[str, Any]:
        """Assess execution risk"""
        # Calculate risk scores
        size_risk = min(1.0, order_size / metrics.average_volume * 10)  # 10% ADV = max risk
        timing_risk = 0.3 if metrics.session in [MarketSession.OPEN_AUCTION, MarketSession.CLOSE_AUCTION] else 0.1
        liquidity_risk = 1.0 - metrics.volume_ratio if metrics.volume_ratio < 1 else 0
        
        overall_risk = (size_risk * 0.5 + timing_risk * 0.2 + liquidity_risk * 0.3)
        
        # Risk level
        if overall_risk > 0.7:
            risk_level = 'high'
        elif overall_risk > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'overall_risk_score': overall_risk,
            'risk_level': risk_level,
            'risk_factors': {
                'size_risk': size_risk,
                'timing_risk': timing_risk,
                'liquidity_risk': liquidity_risk
            },
            'mitigation_strategies': self._get_risk_mitigation(risk_level)
        }
    
    def _get_risk_mitigation(self, risk_level: str) -> List[str]:
        """Get risk mitigation strategies"""
        if risk_level == 'high':
            return [
                "Consider splitting order across multiple days",
                "Use dark pools for 50%+ of volume",
                "Set wide price limits",
                "Monitor for liquidity events",
                "Have cancellation strategy ready"
            ]
        elif risk_level == 'medium':
            return [
                "Use algorithmic execution",
                "Monitor fill quality closely",
                "Be prepared to pause execution",
                "Consider extending time horizon"
            ]
        else:
            return [
                "Proceed with standard execution",
                "Monitor for unusual activity",
                "Adjust if conditions change"
            ]
    
    def _get_alternative_strategies(self, order_size: float,
                                  metrics: LiquidityMetrics,
                                  urgency: str) -> List[Dict[str, Any]]:
        """Get alternative execution strategies"""
        alternatives = []
        
        # Always provide a patient alternative
        patient_rec = self.liquidity_analyzer.recommend_execution_strategy(
            order_size, metrics, 'patient'
        )
        alternatives.append({
            'name': 'Patient Execution',
            'description': 'Minimize impact with extended timeline',
            'expected_cost_bps': patient_rec.expected_cost_bps,
            'time_horizon': patient_rec.time_horizon
        })
        
        # Aggressive alternative if not already urgent
        if urgency != 'immediate':
            aggressive_rec = self.liquidity_analyzer.recommend_execution_strategy(
                order_size, metrics, 'immediate'
            )
            alternatives.append({
                'name': 'Aggressive Execution',
                'description': 'Complete quickly, accept higher impact',
                'expected_cost_bps': aggressive_rec.expected_cost_bps,
                'time_horizon': aggressive_rec.time_horizon
            })
        
        # Opportunistic alternative
        alternatives.append({
            'name': 'Opportunistic Execution',
            'description': 'Wait for liquidity events and volume spikes',
            'expected_cost_bps': order_size / metrics.average_volume * 500,  # Rough estimate
            'time_horizon': 480  # Full day
        })
        
        return alternatives


# Demo function
async def demo_liquidity_prediction():
    """Demonstrate the Liquidity Prediction Agent"""
    agent = LiquidityPredictionAgent()
    
    print("Liquidity Prediction Agent Demo")
    print("="*70)
    
    # Test Case 1: Normal market conditions
    print("\n📊 Case 1: Normal Market Conditions")
    print("-"*50)
    
    market_data_normal = {
        'symbol': 'AAPL',
        'bid': 195.00,
        'ask': 195.02,
        'volume': 50000000,
        'avg_volume': 65000000,
        'price': 195.01
    }
    
    prediction = await agent.predict_liquidity('AAPL', market_data_normal)
    
    print(f"Current Liquidity: {prediction['current_liquidity']['level'].upper()}")
    print(f"  Volume Ratio: {prediction['current_liquidity']['volume_ratio']:.2f}")
    print(f"  Spread: {prediction['current_liquidity']['spread_bps']:.1f} bps")
    
    print(f"\nPredicted Liquidity (30 min): {prediction['predicted_liquidity']['level'].upper()}")
    print(f"  Confidence: {prediction['predicted_liquidity']['confidence']:.1%}")
    
    if prediction['warnings']:
        print(f"\n⚠️ Warnings:")
        for warning in prediction['warnings']:
            print(f"  - {warning}")
    
    # Test Case 2: Large order execution
    print("\n\n📊 Case 2: Large Order Execution Recommendation")
    print("-"*50)
    
    order_size = 5000000  # $5M order
    
    exec_rec = await agent.recommend_execution(
        'AAPL', order_size, 'buy', market_data_normal, urgency='normal'
    )
    
    print(f"Order Size: ${order_size:,.0f} ({order_size/market_data_normal['avg_volume']*100:.1f}% of ADV)")
    print(f"Recommended Strategy: {exec_rec['execution_strategy']['recommended_type'].upper()}")
    print(f"Expected Cost: {exec_rec['execution_strategy']['expected_cost_bps']:.1f} bps")
    
    print(f"\nExecution Schedule:")
    for i, slice_detail in enumerate(exec_rec['execution_schedule'][:3]):
        print(f"  Slice {slice_detail['slice_number']}: ${slice_detail['size']:,.0f} ({slice_detail['percentage']:.1f}%) via {slice_detail['venue']}")
    if len(exec_rec['execution_schedule']) > 3:
        print(f"  ... and {len(exec_rec['execution_schedule'])-3} more slices")
    
    print(f"\nRisk Assessment: {exec_rec['risk_assessment']['risk_level'].upper()}")
    print(f"  Overall Risk Score: {exec_rec['risk_assessment']['overall_risk_score']:.2f}")
    
    # Test Case 3: Low liquidity conditions
    print("\n\n📊 Case 3: Low Liquidity Scenario")
    print("-"*50)
    
    market_data_low = {
        'symbol': 'XYZ',
        'bid': 50.00,
        'ask': 50.10,
        'volume': 500000,
        'avg_volume': 2000000,
        'price': 50.05
    }
    
    low_liq_pred = await agent.predict_liquidity('XYZ', market_data_low)
    
    print(f"Current Liquidity: {low_liq_pred['current_liquidity']['level'].upper()}")
    print(f"  Volume Ratio: {low_liq_pred['current_liquidity']['volume_ratio']:.2f}")
    print(f"  Spread: {low_liq_pred['current_liquidity']['spread_bps']:.1f} bps")
    
    # Small order in low liquidity
    small_order = 50000  # $50k order
    small_exec = await agent.recommend_execution(
        'XYZ', small_order, 'sell', market_data_low, urgency='patient'
    )
    
    print(f"\nSmall Order Execution:")
    print(f"  Strategy: {small_exec['execution_strategy']['recommended_type'].upper()}")
    print(f"  Notes: {'; '.join(small_exec['execution_strategy']['notes'])}")
    
    # Test Case 4: Optimal execution windows
    print("\n\n📊 Case 4: Optimal Execution Windows")
    print("-"*50)
    
    if prediction['optimal_execution_windows']:
        print("Today's Optimal Windows:")
        for window in prediction['optimal_execution_windows'][:3]:
            session_name = window['session'].replace('_', ' ').title()
            print(f"  {session_name}: {window['recommended_for']}")
            print(f"    Liquidity Multiplier: {window['liquidity_multiplier']:.1f}x")


if __name__ == "__main__":
    asyncio.run(demo_liquidity_prediction()) 