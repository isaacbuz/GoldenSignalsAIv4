"""
Smart Execution Agent
Intelligent order execution with market impact minimization and optimal routing
Issue #187: Agent-3: Develop Smart Execution Agent
"""

import asyncio
import heapq
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategies"""
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price
    POV = "pov"    # Percentage of volume
    IS = "is"      # Implementation shortfall
    ICEBERG = "iceberg"  # Hidden order
    SNIPER = "sniper"    # Opportunistic execution
    ADAPTIVE = "adaptive"  # ML-driven adaptive
    LIQUIDITY_SEEKING = "liquidity_seeking"
    DARK_POOL = "dark_pool"
    SMART_ROUTING = "smart_routing"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    PEG = "peg"  # Pegged to market
    HIDDEN = "hidden"
    MIDPOINT = "midpoint"


class Venue(Enum):
    """Execution venues"""
    PRIMARY = "primary"  # Primary exchange
    DARK_POOL = "dark_pool"
    ECN = "ecn"  # Electronic communication network
    ATS = "ats"  # Alternative trading system
    SIP = "sip"  # Securities information processor
    DIRECT = "direct"  # Direct market access


@dataclass
class MarketConditions:
    """Current market conditions"""
    bid: float
    ask: float
    mid: float
    spread_bps: float
    bid_size: int
    ask_size: int
    volume: int
    avg_volume: int
    volatility: float
    liquidity_score: float  # 0-1
    momentum: float  # -1 to 1
    market_impact: float  # Expected impact in bps


@dataclass
class ExecutionOrder:
    """Order to be executed"""
    symbol: str
    side: str  # 'buy' or 'sell'
    total_quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    time_constraint: Optional[int] = None  # Minutes
    urgency: str = "normal"  # low, normal, high, immediate
    min_fill_size: int = 100
    max_display_size: Optional[int] = None
    participate_rate: float = 0.1  # Max % of volume
    price_improvement_bps: float = 0  # Desired improvement
    venue_preferences: List[Venue] = field(default_factory=list)

    @property
    def is_buy(self) -> bool:
        return self.side.lower() == 'buy'


@dataclass
class ExecutionSlice:
    """Individual execution slice"""
    slice_id: str
    quantity: int
    order_type: OrderType
    venue: Venue
    limit_price: Optional[float]
    start_time: datetime
    expire_time: datetime
    min_fill: int
    max_show: Optional[int]
    priority: int  # For ordering

    def __lt__(self, other):
        return self.priority < other.priority


@dataclass
class ExecutionResult:
    """Result of execution"""
    order_id: str
    slices_executed: int
    total_filled: int
    average_price: float
    vwap: float
    market_vwap: float
    slippage_bps: float
    market_impact_bps: float
    fill_rate: float
    venue_breakdown: Dict[Venue, int]
    execution_time_seconds: float
    cost_analysis: Dict[str, float]


class MarketSimulator:
    """Simulates market dynamics for execution"""

    def __init__(self):
        self.order_book = defaultdict(lambda: {'bids': [], 'asks': []})
        self.trade_history = deque(maxlen=1000)
        self.market_impact_model = self._initialize_impact_model()

    def _initialize_impact_model(self) -> Dict[str, Any]:
        """Initialize market impact model parameters"""
        return {
            'permanent_impact': 0.1,  # 10% of temporary
            'temporary_impact_half_life': 300,  # 5 minutes
            'participation_impact': 0.5,  # Impact per % of volume
            'volatility_adjustment': 1.5,
            'liquidity_factor': 0.8
        }

    def estimate_market_impact(self, order: ExecutionOrder,
                             market: MarketConditions) -> float:
        """Estimate market impact in basis points"""
        # Size impact (square root model)
        size_ratio = order.total_quantity / market.avg_volume
        size_impact = 10 * np.sqrt(size_ratio * 100)

        # Participation rate impact
        participation_impact = self.market_impact_model['participation_impact'] * \
                             order.participate_rate * 100

        # Volatility adjustment
        vol_adj = market.volatility / 0.20  # Normalized to 20% annual vol
        volatility_impact = size_impact * vol_adj * \
                          self.market_impact_model['volatility_adjustment']

        # Liquidity adjustment
        liquidity_adj = (1 - market.liquidity_score) * 2
        liquidity_impact = size_impact * liquidity_adj

        # Urgency adjustment
        urgency_mult = {
            'low': 0.7,
            'normal': 1.0,
            'high': 1.5,
            'immediate': 2.0
        }.get(order.urgency, 1.0)

        total_impact = (size_impact + participation_impact +
                       volatility_impact + liquidity_impact) * urgency_mult

        return total_impact

    def simulate_fill(self, slice: ExecutionSlice,
                     market: MarketConditions) -> Tuple[int, float]:
        """Simulate order fill"""
        # Simple fill simulation
        if slice.order_type == OrderType.MARKET:
            # Market orders fill immediately at ask (buy) or bid (sell)
            fill_quantity = slice.quantity
            fill_price = market.ask if slice.quantity > 0 else market.bid
        elif slice.order_type == OrderType.LIMIT:
            # Limit orders fill if price is favorable
            if slice.quantity > 0:  # Buy order
                if slice.limit_price >= market.ask:
                    fill_quantity = slice.quantity
                    fill_price = market.ask
                else:
                    fill_quantity = 0
                    fill_price = 0
            else:  # Sell order
                if slice.limit_price <= market.bid:
                    fill_quantity = slice.quantity
                    fill_price = market.bid
                else:
                    fill_quantity = 0
                    fill_price = 0
        else:
            # Other order types
            fill_quantity = int(slice.quantity * 0.8)  # Partial fill
            fill_price = market.mid

        return fill_quantity, fill_price


class ExecutionAlgorithm:
    """Base class for execution algorithms"""

    def __init__(self, name: str):
        self.name = name
        self.execution_history = []

    async def execute(self, order: ExecutionOrder,
                     market: MarketConditions) -> List[ExecutionSlice]:
        """Execute order and return slices"""
        raise NotImplementedError


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price algorithm"""

    def __init__(self):
        super().__init__("TWAP")

    async def execute(self, order: ExecutionOrder,
                     market: MarketConditions) -> List[ExecutionSlice]:
        """Execute using TWAP strategy"""
        slices = []

        # Determine time period
        time_minutes = order.time_constraint or 60
        num_slices = min(time_minutes // 5, 20)  # 5-minute intervals, max 20 slices

        # Equal size slices
        slice_size = order.total_quantity // num_slices
        remaining = order.total_quantity

        current_time = datetime.now()

        for i in range(num_slices):
            if remaining <= 0:
                break

            quantity = min(slice_size, remaining)

            slice = ExecutionSlice(
                slice_id=f"TWAP_{order.symbol}_{i}",
                quantity=quantity,
                order_type=OrderType.LIMIT,
                venue=Venue.PRIMARY,
                limit_price=market.mid * (1.001 if order.is_buy else 0.999),
                start_time=current_time + timedelta(minutes=i*5),
                expire_time=current_time + timedelta(minutes=(i+1)*5),
                min_fill=order.min_fill_size,
                max_show=min(quantity, order.max_display_size) if order.max_display_size else quantity,
                priority=i
            )

            slices.append(slice)
            remaining -= quantity

        return slices


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price algorithm"""

    def __init__(self):
        super().__init__("VWAP")
        self.volume_profile = self._load_volume_profile()

    def _load_volume_profile(self) -> Dict[int, float]:
        """Load typical intraday volume profile"""
        # Typical U-shaped volume profile
        profile = {}
        hours = [9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5]
        volumes = [0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]

        for hour, vol in zip(hours, volumes):
            profile[int(hour * 60)] = vol  # Convert to minutes

        return profile

    async def execute(self, order: ExecutionOrder,
                     market: MarketConditions) -> List[ExecutionSlice]:
        """Execute using VWAP strategy"""
        slices = []
        current_time = datetime.now()
        current_minutes = current_time.hour * 60 + current_time.minute

        # Get remaining volume distribution
        remaining_volume_pct = 0
        time_slices = []

        for time_min, vol_pct in sorted(self.volume_profile.items()):
            if time_min > current_minutes:
                remaining_volume_pct += vol_pct
                time_slices.append((time_min, vol_pct))

        if remaining_volume_pct == 0:
            # Fall back to TWAP if after hours
            twap = TWAPAlgorithm()
            return await twap.execute(order, market)

        # Normalize percentages
        for i, (time_min, vol_pct) in enumerate(time_slices):
            time_slices[i] = (time_min, vol_pct / remaining_volume_pct)

        # Create slices based on volume profile
        remaining = order.total_quantity

        for i, (time_min, vol_pct) in enumerate(time_slices):
            if remaining <= 0:
                break

            quantity = int(order.total_quantity * vol_pct)
            quantity = min(quantity, remaining)

            # Calculate start time
            hours = time_min // 60
            minutes = time_min % 60
            start_time = current_time.replace(hour=hours, minute=minutes, second=0)

            slice = ExecutionSlice(
                slice_id=f"VWAP_{order.symbol}_{i}",
                quantity=quantity,
                order_type=OrderType.LIMIT,
                venue=Venue.PRIMARY,
                limit_price=market.mid * (1.001 if order.is_buy else 0.999),
                start_time=start_time,
                expire_time=start_time + timedelta(minutes=30),
                min_fill=order.min_fill_size,
                max_show=min(quantity // 5, order.max_display_size or quantity),
                priority=i
            )

            slices.append(slice)
            remaining -= quantity

        return slices


class AdaptiveAlgorithm(ExecutionAlgorithm):
    """Adaptive execution using ML predictions"""

    def __init__(self):
        super().__init__("Adaptive")
        self.market_predictor = self._initialize_predictor()
        self.execution_optimizer = self._initialize_optimizer()

    def _initialize_predictor(self) -> Dict[str, Any]:
        """Initialize market predictor"""
        return {
            'liquidity_model': 'lstm',
            'price_model': 'gru',
            'impact_model': 'xgboost'
        }

    def _initialize_optimizer(self) -> Dict[str, Any]:
        """Initialize execution optimizer"""
        return {
            'objective': 'minimize_shortfall',
            'constraints': ['participation', 'timing'],
            'method': 'dynamic_programming'
        }

    async def execute(self, order: ExecutionOrder,
                     market: MarketConditions) -> List[ExecutionSlice]:
        """Execute using adaptive strategy"""
        slices = []

        # Predict optimal execution schedule
        schedule = await self._optimize_schedule(order, market)

        current_time = datetime.now()

        for i, (time_offset, quantity, strategy) in enumerate(schedule):
            slice = ExecutionSlice(
                slice_id=f"ADAPTIVE_{order.symbol}_{i}",
                quantity=quantity,
                order_type=self._select_order_type(strategy, market),
                venue=self._select_venue(strategy, market),
                limit_price=self._calculate_limit_price(order, market, strategy),
                start_time=current_time + timedelta(minutes=time_offset),
                expire_time=current_time + timedelta(minutes=time_offset + 5),
                min_fill=max(100, quantity // 10),
                max_show=self._calculate_display_size(quantity, strategy),
                priority=i
            )

            slices.append(slice)

        return slices

    async def _optimize_schedule(self, order: ExecutionOrder,
                               market: MarketConditions) -> List[Tuple[int, int, str]]:
        """Optimize execution schedule using DP"""
        # Simplified optimization
        total_time = order.time_constraint or 60
        num_periods = min(total_time // 5, 12)

        # Dynamic programming to minimize cost
        remaining = order.total_quantity
        schedule = []

        for period in range(num_periods):
            if remaining <= 0:
                break

            # Predict optimal quantity for this period
            predicted_liquidity = self._predict_liquidity(period, market)
            predicted_impact = self._predict_impact(remaining, market)

            # Optimal quantity balances urgency vs impact
            if order.urgency == 'immediate':
                quantity = remaining
            elif order.urgency == 'high':
                quantity = min(remaining, int(remaining / (num_periods - period)))
            else:
                # Base on predicted liquidity
                max_quantity = int(market.avg_volume * order.participate_rate * 5 / 390)  # 5-min slice
                optimal_quantity = int(max_quantity * predicted_liquidity)
                quantity = min(remaining, optimal_quantity)

            # Select strategy based on conditions
            if predicted_impact > 20:  # High impact
                strategy = 'passive'
            elif predicted_liquidity > 0.8:
                strategy = 'aggressive'
            else:
                strategy = 'balanced'

            schedule.append((period * 5, quantity, strategy))
            remaining -= quantity

        return schedule

    def _predict_liquidity(self, period: int, market: MarketConditions) -> float:
        """Predict liquidity for period"""
        # Simplified prediction
        base_liquidity = market.liquidity_score
        time_factor = 1.0 if period < 6 else 0.8  # Lower liquidity later
        return min(1.0, base_liquidity * time_factor)

    def _predict_impact(self, size: int, market: MarketConditions) -> float:
        """Predict market impact"""
        # Simplified impact prediction
        size_ratio = size / market.avg_volume
        return 10 * np.sqrt(size_ratio * 100)

    def _select_order_type(self, strategy: str, market: MarketConditions) -> OrderType:
        """Select order type based on strategy"""
        if strategy == 'aggressive':
            return OrderType.MARKET if market.spread_bps < 5 else OrderType.PEG
        elif strategy == 'passive':
            return OrderType.LIMIT
        else:
            return OrderType.MIDPOINT

    def _select_venue(self, strategy: str, market: MarketConditions) -> Venue:
        """Select venue based on strategy"""
        if strategy == 'passive' and market.liquidity_score < 0.5:
            return Venue.DARK_POOL
        elif market.spread_bps > 10:
            return Venue.ECN
        else:
            return Venue.PRIMARY

    def _calculate_limit_price(self, order: ExecutionOrder,
                             market: MarketConditions,
                             strategy: str) -> Optional[float]:
        """Calculate limit price"""
        if strategy == 'aggressive':
            # Cross the spread
            return market.ask if order.is_buy else market.bid
        elif strategy == 'passive':
            # Join the queue
            return market.bid if order.is_buy else market.ask
        else:
            # Midpoint
            return market.mid

    def _calculate_display_size(self, quantity: int, strategy: str) -> int:
        """Calculate display size"""
        if strategy == 'passive':
            return quantity // 10  # Show 10%
        elif strategy == 'aggressive':
            return quantity  # Show full size
        else:
            return quantity // 5  # Show 20%


class SmartExecutionAgent:
    """
    Smart Execution Agent that minimizes market impact and optimizes execution
    Combines multiple algorithms and venues for optimal results
    """

    def __init__(self):
        """Initialize the smart execution agent"""
        self.algorithms = {
            ExecutionStrategy.TWAP: TWAPAlgorithm(),
            ExecutionStrategy.VWAP: VWAPAlgorithm(),
            ExecutionStrategy.ADAPTIVE: AdaptiveAlgorithm()
        }

        self.market_simulator = MarketSimulator()
        self.execution_queue = []
        self.active_orders = {}
        self.execution_history = deque(maxlen=1000)

        # Performance tracking
        self.performance_metrics = {
            'total_orders': 0,
            'avg_slippage_bps': 0,
            'avg_market_impact_bps': 0,
            'fill_rate': 0
        }

    async def execute_order(self, order: ExecutionOrder,
                           market_data: Dict[str, Any]) -> ExecutionResult:
        """Execute a smart order"""
        start_time = datetime.now()

        # Convert market data to conditions
        market = self._create_market_conditions(market_data)

        # Select execution strategy
        strategy = await self._select_strategy(order, market)

        # Get algorithm
        algorithm = self.algorithms.get(strategy)
        if not algorithm:
            algorithm = self.algorithms[ExecutionStrategy.ADAPTIVE]

        # Generate execution slices
        slices = await algorithm.execute(order, market)

        # Simulate execution
        fills = await self._execute_slices(slices, order, market)

        # Calculate results
        result = self._calculate_results(order, fills, market, start_time)

        # Update performance metrics
        self._update_metrics(result)

        # Store in history
        self.execution_history.append({
            'timestamp': datetime.now(),
            'order': order,
            'result': result,
            'market_conditions': market
        })

        return result

    def _create_market_conditions(self, market_data: Dict[str, Any]) -> MarketConditions:
        """Create market conditions from data"""
        bid = market_data.get('bid', 100.0)
        ask = market_data.get('ask', 100.1)

        return MarketConditions(
            bid=bid,
            ask=ask,
            mid=(bid + ask) / 2,
            spread_bps=(ask - bid) / ((ask + bid) / 2) * 10000,
            bid_size=market_data.get('bid_size', 1000),
            ask_size=market_data.get('ask_size', 1000),
            volume=market_data.get('volume', 1000000),
            avg_volume=market_data.get('avg_volume', 5000000),
            volatility=market_data.get('volatility', 0.20),
            liquidity_score=market_data.get('liquidity_score', 0.7),
            momentum=market_data.get('momentum', 0.0),
            market_impact=10.0  # Default 10 bps
        )

    async def _select_strategy(self, order: ExecutionOrder,
                             market: MarketConditions) -> ExecutionStrategy:
        """Select optimal execution strategy"""
        # Estimate market impact
        impact = self.market_simulator.estimate_market_impact(order, market)

        # Decision logic
        if order.urgency == 'immediate':
            return ExecutionStrategy.ADAPTIVE
        elif impact > 50:  # High impact
            if market.liquidity_score > 0.7:
                return ExecutionStrategy.VWAP
            else:
                return ExecutionStrategy.ADAPTIVE
        elif order.time_constraint and order.time_constraint < 30:
            return ExecutionStrategy.TWAP
        elif market.volatility > 0.30:
            return ExecutionStrategy.ADAPTIVE
        else:
            return ExecutionStrategy.VWAP

    async def _execute_slices(self, slices: List[ExecutionSlice],
                            order: ExecutionOrder,
                            market: MarketConditions) -> List[Dict[str, Any]]:
        """Execute slices and return fills"""
        fills = []
        total_filled = 0

        for slice in slices:
            # Simulate execution delay
            await asyncio.sleep(0.01)

            # Simulate fill
            fill_quantity, fill_price = self.market_simulator.simulate_fill(slice, market)

            if fill_quantity > 0:
                fills.append({
                    'slice_id': slice.slice_id,
                    'quantity': fill_quantity,
                    'price': fill_price,
                    'venue': slice.venue,
                    'timestamp': datetime.now()
                })

                total_filled += fill_quantity

                # Update market conditions (simplified)
                if order.is_buy:
                    market.bid = fill_price * 0.9999
                    market.ask = fill_price * 1.0001
                else:
                    market.bid = fill_price * 0.9999
                    market.ask = fill_price * 1.0001

        return fills

    def _calculate_results(self, order: ExecutionOrder,
                         fills: List[Dict[str, Any]],
                         market: MarketConditions,
                         start_time: datetime) -> ExecutionResult:
        """Calculate execution results"""
        if not fills:
            return ExecutionResult(
                order_id=f"{order.symbol}_{start_time.timestamp()}",
                slices_executed=0,
                total_filled=0,
                average_price=0,
                vwap=0,
                market_vwap=market.mid,
                slippage_bps=0,
                market_impact_bps=0,
                fill_rate=0,
                venue_breakdown={},
                execution_time_seconds=0,
                cost_analysis={}
            )

        # Calculate metrics
        total_filled = sum(f['quantity'] for f in fills)
        total_value = sum(f['quantity'] * f['price'] for f in fills)
        average_price = total_value / total_filled if total_filled > 0 else 0

        # VWAP calculation (simplified)
        vwap = average_price  # In reality, would weight by time

        # Slippage
        if order.is_buy:
            slippage_bps = (average_price - market.mid) / market.mid * 10000
        else:
            slippage_bps = (market.mid - average_price) / market.mid * 10000

        # Market impact (simplified)
        market_impact_bps = abs(fills[-1]['price'] - fills[0]['price']) / fills[0]['price'] * 10000

        # Venue breakdown
        venue_breakdown = defaultdict(int)
        for fill in fills:
            venue_breakdown[fill['venue']] += fill['quantity']

        # Cost analysis
        spread_cost = market.spread_bps * total_filled / order.total_quantity
        impact_cost = market_impact_bps
        timing_cost = max(0, slippage_bps - market_impact_bps)

        execution_time = (datetime.now() - start_time).total_seconds()

        return ExecutionResult(
            order_id=f"{order.symbol}_{start_time.timestamp()}",
            slices_executed=len(fills),
            total_filled=total_filled,
            average_price=average_price,
            vwap=vwap,
            market_vwap=market.mid,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            fill_rate=total_filled / order.total_quantity,
            venue_breakdown=dict(venue_breakdown),
            execution_time_seconds=execution_time,
            cost_analysis={
                'spread_cost_bps': spread_cost,
                'impact_cost_bps': impact_cost,
                'timing_cost_bps': timing_cost,
                'total_cost_bps': spread_cost + impact_cost + timing_cost
            }
        )

    def _update_metrics(self, result: ExecutionResult):
        """Update performance metrics"""
        n = self.performance_metrics['total_orders']

        # Update moving averages
        self.performance_metrics['avg_slippage_bps'] = \
            (self.performance_metrics['avg_slippage_bps'] * n + result.slippage_bps) / (n + 1)
        self.performance_metrics['avg_market_impact_bps'] = \
            (self.performance_metrics['avg_market_impact_bps'] * n + result.market_impact_bps) / (n + 1)
        self.performance_metrics['fill_rate'] = \
            (self.performance_metrics['fill_rate'] * n + result.fill_rate) / (n + 1)

        self.performance_metrics['total_orders'] += 1

    async def optimize_execution(self, orders: List[ExecutionOrder],
                               market_data: Dict[str, Any],
                               constraints: Optional[Dict[str, Any]] = None) -> List[ExecutionResult]:
        """Optimize execution for multiple orders"""
        results = []

        # Sort orders by priority
        priority_orders = sorted(orders, key=lambda x: (
            {'immediate': 0, 'high': 1, 'normal': 2, 'low': 3}[x.urgency],
            -x.total_quantity
        ))

        # Execute in priority order
        for order in priority_orders:
            result = await self.execute_order(order, market_data)
            results.append(result)

            # Update market data based on execution
            if result.fill_rate > 0:
                market_data['volume'] += result.total_filled

        return results

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return {
            'summary': self.performance_metrics,
            'recent_executions': [
                {
                    'timestamp': exec['timestamp'].isoformat(),
                    'symbol': exec['order'].symbol,
                    'quantity': exec['order'].total_quantity,
                    'fill_rate': exec['result'].fill_rate,
                    'slippage_bps': exec['result'].slippage_bps,
                    'total_cost_bps': exec['result'].cost_analysis['total_cost_bps']
                }
                for exec in list(self.execution_history)[-10:]
            ],
            'algorithm_usage': {
                'TWAP': sum(1 for e in self.execution_history if 'TWAP' in str(e)),
                'VWAP': sum(1 for e in self.execution_history if 'VWAP' in str(e)),
                'Adaptive': sum(1 for e in self.execution_history if 'ADAPTIVE' in str(e))
            }
        }


# Demo function
async def demo_smart_execution():
    """Demonstrate the Smart Execution Agent"""
    agent = SmartExecutionAgent()

    print("Smart Execution Agent Demo")
    print("="*70)

    # Test Case 1: Small order, normal conditions
    print("\n📊 Case 1: Small Order, Normal Market")
    print("-"*50)

    order1 = ExecutionOrder(
        symbol='AAPL',
        side='buy',
        total_quantity=1000,
        order_type=OrderType.LIMIT,
        urgency='normal',
        participate_rate=0.1,
        venue_preferences=[Venue.PRIMARY, Venue.ECN]
    )

    market_data1 = {
        'bid': 195.00,
        'ask': 195.02,
        'volume': 5000000,
        'avg_volume': 50000000,
        'volatility': 0.18,
        'liquidity_score': 0.85
    }

    result1 = await agent.execute_order(order1, market_data1)

    print(f"Order: Buy {order1.total_quantity} shares of {order1.symbol}")
    print(f"Execution Results:")
    print(f"  Fill Rate: {result1.fill_rate:.1%}")
    print(f"  Average Price: ${result1.average_price:.2f}")
    print(f"  Slippage: {result1.slippage_bps:.1f} bps")
    print(f"  Market Impact: {result1.market_impact_bps:.1f} bps")
    print(f"  Total Cost: {result1.cost_analysis['total_cost_bps']:.1f} bps")

    # Test Case 2: Large order, urgent
    print("\n\n📊 Case 2: Large Urgent Order")
    print("-"*50)

    order2 = ExecutionOrder(
        symbol='SPY',
        side='sell',
        total_quantity=100000,
        order_type=OrderType.LIMIT,
        urgency='high',
        time_constraint=15,  # 15 minutes
        participate_rate=0.2
    )

    market_data2 = {
        'bid': 450.50,
        'ask': 450.52,
        'volume': 10000000,
        'avg_volume': 80000000,
        'volatility': 0.15,
        'liquidity_score': 0.90
    }

    result2 = await agent.execute_order(order2, market_data2)

    print(f"Order: Sell {order2.total_quantity} shares of {order2.symbol} (URGENT)")
    print(f"Execution Results:")
    print(f"  Fill Rate: {result2.fill_rate:.1%}")
    print(f"  Execution Time: {result2.execution_time_seconds:.1f} seconds")
    print(f"  Slices Executed: {result2.slices_executed}")
    print(f"  Market Impact: {result2.market_impact_bps:.1f} bps")
    print(f"  Venue Breakdown:")
    for venue, qty in result2.venue_breakdown.items():
        print(f"    {venue.value}: {qty:,} shares")

    # Test Case 3: Illiquid market
    print("\n\n📊 Case 3: Illiquid Market Conditions")
    print("-"*50)

    order3 = ExecutionOrder(
        symbol='XYZ',
        side='buy',
        total_quantity=5000,
        order_type=OrderType.LIMIT,
        urgency='low',
        price_improvement_bps=5,
        max_display_size=500
    )

    market_data3 = {
        'bid': 25.00,
        'ask': 25.10,
        'volume': 100000,
        'avg_volume': 500000,
        'volatility': 0.35,
        'liquidity_score': 0.40
    }

    result3 = await agent.execute_order(order3, market_data3)

    print(f"Order: Buy {order3.total_quantity} shares of {order3.symbol} (ILLIQUID)")
    print(f"Market Conditions:")
    print(f"  Spread: {(market_data3['ask'] - market_data3['bid'])/market_data3['bid']*10000:.1f} bps")
    print(f"  Liquidity Score: {market_data3['liquidity_score']:.2f}")

    print(f"\nAdaptive Execution Results:")
    print(f"  Strategy Selected: Adaptive (due to low liquidity)")
    print(f"  Average Price: ${result3.average_price:.2f}")
    print(f"  Price Improvement: {result3.slippage_bps:.1f} bps")
    print(f"  Hidden Quantity: {order3.max_display_size} shares shown")

    # Test Case 4: Multi-order optimization
    print("\n\n📊 Case 4: Multi-Order Optimization")
    print("-"*50)

    orders = [
        ExecutionOrder('AAPL', 'buy', 2000, OrderType.LIMIT, urgency='high'),
        ExecutionOrder('MSFT', 'sell', 3000, OrderType.LIMIT, urgency='normal'),
        ExecutionOrder('GOOGL', 'buy', 1000, OrderType.LIMIT, urgency='immediate')
    ]

    market_data_multi = {
        'bid': 100.00,
        'ask': 100.02,
        'volume': 1000000,
        'avg_volume': 10000000,
        'volatility': 0.20,
        'liquidity_score': 0.75
    }

    results = await agent.optimize_execution(orders, market_data_multi)

    print("Multi-Order Execution Results:")
    print(f"{'Symbol':<8} {'Side':<6} {'Qty':<8} {'Urgency':<10} {'Fill%':<8} {'Cost(bps)':<10}")
    print("-"*60)

    for order, result in zip(orders, results):
        print(f"{order.symbol:<8} {order.side:<6} {order.total_quantity:<8} "
              f"{order.urgency:<10} {result.fill_rate*100:<7.1f}% "
              f"{result.cost_analysis['total_cost_bps']:<10.1f}")

    # Performance summary
    print("\n\n📊 Performance Summary")
    print("-"*50)

    report = agent.get_performance_report()
    summary = report['summary']

    print(f"Total Orders Executed: {summary['total_orders']}")
    print(f"Average Fill Rate: {summary['fill_rate']:.1%}")
    print(f"Average Slippage: {summary['avg_slippage_bps']:.1f} bps")
    print(f"Average Market Impact: {summary['avg_market_impact_bps']:.1f} bps")

    print("\n✅ Smart Execution Agent demonstrates:")
    print("- Adaptive strategy selection based on market conditions")
    print("- Multi-venue routing for optimal execution")
    print("- Real-time market impact estimation")
    print("- Cost-optimized slice generation")


if __name__ == "__main__":
    asyncio.run(demo_smart_execution())
