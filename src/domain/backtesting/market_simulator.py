"""
Market Microstructure Simulator - Realistic market simulation
Simulates order books, bid-ask spreads, market impact, and execution dynamics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the simulator"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    avg_fill_price: float = 0
    fills: List[Dict] = field(default_factory=list)
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity


@dataclass
class OrderBookLevel:
    """Single level in the order book"""
    price: float
    quantity: float
    orders: int = 1  # Number of orders at this level


@dataclass
class OrderBook:
    """Simulated order book with Level 2 data"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    last_price: float = 0
    last_size: float = 0
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.last_price


class MarketMicrostructureSimulator:
    """
    Simulates realistic market microstructure including:
    - Order book dynamics
    - Bid-ask spreads
    - Market impact
    - Execution latency
    - Partial fills
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Market parameters
        self.base_spread_bps = self.config.get('base_spread_bps', 2)  # 2 basis points
        self.spread_volatility = self.config.get('spread_volatility', 0.5)
        self.market_impact_factor = self.config.get('market_impact_factor', 0.1)
        self.latency_ms = self.config.get('latency_ms', 50)
        
        # Order book parameters
        self.book_depth = self.config.get('book_depth', 10)
        self.level_spacing_bps = self.config.get('level_spacing_bps', 1)
        
        # State
        self.order_books: Dict[str, OrderBook] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.market_trades: List[Dict] = []
        
    def simulate_order_book(
        self,
        symbol: str,
        mid_price: float,
        volume: float,
        volatility: float,
        timestamp: datetime
    ) -> OrderBook:
        """
        Generate a realistic order book based on market conditions
        
        Args:
            symbol: Stock symbol
            mid_price: Current mid price
            volume: Recent trading volume
            volatility: Current volatility
            timestamp: Current time
        """
        # Calculate dynamic spread based on volatility and volume
        volume_factor = np.clip(1000000 / volume, 0.5, 2.0)  # Lower volume = wider spread
        volatility_factor = 1 + volatility * 10  # Higher volatility = wider spread
        
        spread_bps = self.base_spread_bps * volume_factor * volatility_factor
        spread_amount = mid_price * spread_bps / 10000
        
        # Add randomness to spread
        spread_amount *= (1 + np.random.normal(0, self.spread_volatility))
        spread_amount = max(0.01, spread_amount)  # Minimum 1 cent spread
        
        # Generate order book levels
        bids = []
        asks = []
        
        # Best bid/ask
        best_bid = mid_price - spread_amount / 2
        best_ask = mid_price + spread_amount / 2
        
        # Generate book depth with realistic size distribution
        for i in range(self.book_depth):
            # Price levels
            bid_price = best_bid - (i * mid_price * self.level_spacing_bps / 10000)
            ask_price = best_ask + (i * mid_price * self.level_spacing_bps / 10000)
            
            # Size distribution - more liquidity at better prices
            size_multiplier = np.exp(-i * 0.3)  # Exponential decay
            base_size = volume / 1000  # Base on recent volume
            
            bid_size = base_size * size_multiplier * np.random.lognormal(0, 0.5)
            ask_size = base_size * size_multiplier * np.random.lognormal(0, 0.5)
            
            # Number of orders at each level
            num_orders = max(1, int(np.random.poisson(3 - i * 0.2)))
            
            bids.append(OrderBookLevel(
                price=round(bid_price, 2),
                quantity=round(bid_size, 0),
                orders=num_orders
            ))
            
            asks.append(OrderBookLevel(
                price=round(ask_price, 2),
                quantity=round(ask_size, 0),
                orders=num_orders
            ))
        
        order_book = OrderBook(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            last_price=mid_price,
            last_size=volume / 100
        )
        
        self.order_books[symbol] = order_book
        return order_book
    
    def estimate_market_impact(
        self,
        order: Order,
        order_book: OrderBook,
        avg_daily_volume: float
    ) -> float:
        """
        Estimate market impact using Kyle's lambda model
        
        Args:
            order: The order to execute
            order_book: Current order book
            avg_daily_volume: Average daily volume
            
        Returns:
            Estimated price impact in percentage
        """
        # Calculate order size as percentage of ADV
        order_size_pct = order.quantity / avg_daily_volume
        
        # Kyle's lambda - temporary impact
        # Impact = lambda * sqrt(order_size / ADV)
        temporary_impact = self.market_impact_factor * np.sqrt(order_size_pct)
        
        # Permanent impact (usually 50-70% of temporary)
        permanent_impact = temporary_impact * 0.6
        
        # Adjust for order book imbalance
        if order_book.best_bid and order_book.best_ask:
            total_bid_size = sum(level.quantity for level in order_book.bids[:3])
            total_ask_size = sum(level.quantity for level in order_book.asks[:3])
            
            imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
            
            # If buying into weak ask side, higher impact
            if order.side == OrderSide.BUY and imbalance > 0:
                temporary_impact *= (1 + abs(imbalance))
            elif order.side == OrderSide.SELL and imbalance < 0:
                temporary_impact *= (1 + abs(imbalance))
        
        return temporary_impact
    
    async def execute_order(
        self,
        order: Order,
        order_book: OrderBook,
        market_data: Dict[str, Any]
    ) -> Order:
        """
        Simulate realistic order execution
        
        Args:
            order: Order to execute
            order_book: Current order book
            market_data: Additional market data (volume, volatility, etc.)
            
        Returns:
            Executed order with fill information
        """
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Check order validity
        if order.order_type == OrderType.MARKET:
            executed_order = await self._execute_market_order(order, order_book, market_data)
        elif order.order_type == OrderType.LIMIT:
            executed_order = await self._execute_limit_order(order, order_book, market_data)
        elif order.order_type == OrderType.STOP:
            executed_order = await self._execute_stop_order(order, order_book, market_data)
        else:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Unsupported order type: {order.order_type}")
            return order
        
        # Record order
        self.order_history.append(executed_order)
        
        # Update order book after execution
        self._update_order_book_after_trade(order_book, executed_order)
        
        return executed_order
    
    async def _execute_market_order(
        self,
        order: Order,
        order_book: OrderBook,
        market_data: Dict[str, Any]
    ) -> Order:
        """Execute a market order with realistic fills"""
        remaining = order.quantity
        total_cost = 0
        fills = []
        
        # Get relevant order book side
        if order.side == OrderSide.BUY:
            levels = order_book.asks
        else:
            levels = order_book.bids
        
        # Calculate market impact
        impact_pct = self.estimate_market_impact(
            order, order_book, 
            market_data.get('avg_daily_volume', 1000000)
        )
        
        # Walk through order book levels
        for i, level in enumerate(levels):
            if remaining <= 0:
                break
            
            # Apply market impact to price
            if order.side == OrderSide.BUY:
                fill_price = level.price * (1 + impact_pct * (i + 1) / len(levels))
            else:
                fill_price = level.price * (1 - impact_pct * (i + 1) / len(levels))
            
            # Determine fill size (with some randomness for realism)
            available = level.quantity * np.random.uniform(0.8, 1.0)  # 80-100% available
            fill_size = min(remaining, available)
            
            if fill_size > 0:
                fills.append({
                    'price': round(fill_price, 2),
                    'quantity': round(fill_size, 0),
                    'timestamp': datetime.now()
                })
                
                total_cost += fill_price * fill_size
                remaining -= fill_size
                order.filled_quantity += fill_size
        
        # Calculate average fill price
        if order.filled_quantity > 0:
            order.avg_fill_price = total_cost / order.filled_quantity
            order.fills = fills
            
            if remaining <= 0:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIAL
        else:
            order.status = OrderStatus.REJECTED
        
        return order
    
    async def _execute_limit_order(
        self,
        order: Order,
        order_book: OrderBook,
        market_data: Dict[str, Any]
    ) -> Order:
        """Execute a limit order"""
        # For limit orders, check if price is available
        if order.side == OrderSide.BUY:
            if order_book.best_ask and order.price >= order_book.best_ask:
                # Marketable limit order - execute immediately
                return await self._execute_market_order(order, order_book, market_data)
            else:
                # Add to order book
                order.status = OrderStatus.PENDING
                self.pending_orders[order.order_id] = order
        else:  # SELL
            if order_book.best_bid and order.price <= order_book.best_bid:
                # Marketable limit order
                return await self._execute_market_order(order, order_book, market_data)
            else:
                # Add to order book
                order.status = OrderStatus.PENDING
                self.pending_orders[order.order_id] = order
        
        return order
    
    async def _execute_stop_order(
        self,
        order: Order,
        order_book: OrderBook,
        market_data: Dict[str, Any]
    ) -> Order:
        """Execute a stop order"""
        current_price = order_book.last_price
        
        # Check if stop is triggered
        if order.side == OrderSide.BUY and current_price >= order.stop_price:
            # Convert to market order
            order.order_type = OrderType.MARKET
            return await self._execute_market_order(order, order_book, market_data)
        elif order.side == OrderSide.SELL and current_price <= order.stop_price:
            # Convert to market order
            order.order_type = OrderType.MARKET
            return await self._execute_market_order(order, order_book, market_data)
        else:
            # Not triggered yet
            order.status = OrderStatus.PENDING
            self.pending_orders[order.order_id] = order
        
        return order
    
    def _update_order_book_after_trade(self, order_book: OrderBook, order: Order):
        """Update order book after a trade execution"""
        if order.status != OrderStatus.FILLED and order.status != OrderStatus.PARTIAL:
            return
        
        # Update last price and size
        if order.fills:
            last_fill = order.fills[-1]
            order_book.last_price = last_fill['price']
            order_book.last_size = last_fill['quantity']
        
        # Remove liquidity from order book
        if order.side == OrderSide.BUY:
            levels = order_book.asks
        else:
            levels = order_book.bids
        
        remaining_to_remove = order.filled_quantity
        levels_to_remove = []
        
        for i, level in enumerate(levels):
            if remaining_to_remove <= 0:
                break
            
            if level.quantity <= remaining_to_remove:
                levels_to_remove.append(i)
                remaining_to_remove -= level.quantity
            else:
                level.quantity -= remaining_to_remove
                remaining_to_remove = 0
        
        # Remove depleted levels
        for i in reversed(levels_to_remove):
            levels.pop(i)
    
    def simulate_queue_position(
        self,
        order: Order,
        order_book: OrderBook,
        same_price_orders: int
    ) -> float:
        """
        Estimate queue position for limit orders
        
        Returns:
            Probability of fill based on queue position (0-1)
        """
        # Find the level where our order would sit
        if order.side == OrderSide.BUY:
            levels = order_book.bids
        else:
            levels = order_book.asks
        
        # Find matching price level
        for level in levels:
            if abs(level.price - order.price) < 0.01:
                # Estimate our position in queue
                # Assume uniform distribution of orders
                position_pct = (same_price_orders + 1) / (level.orders + same_price_orders + 1)
                
                # Probability of fill decreases with position
                fill_probability = 1 - position_pct
                
                # Adjust for order size
                size_factor = min(1, order.quantity / level.quantity)
                fill_probability *= (1 - size_factor * 0.5)  # Large orders less likely to fill
                
                return fill_probability
        
        return 0.5  # Default if price level not found
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get execution quality analytics"""
        if not self.order_history:
            return {}
        
        filled_orders = [o for o in self.order_history if o.status == OrderStatus.FILLED]
        
        if not filled_orders:
            return {'message': 'No filled orders'}
        
        # Calculate metrics
        total_orders = len(self.order_history)
        fill_rate = len(filled_orders) / total_orders
        
        # Slippage analysis
        market_orders = [o for o in filled_orders if o.order_type == OrderType.MARKET]
        if market_orders:
            slippages = []
            for order in market_orders:
                if order.symbol in self.order_books:
                    book = self.order_books[order.symbol]
                    if order.side == OrderSide.BUY:
                        expected_price = book.best_ask
                    else:
                        expected_price = book.best_bid
                    
                    if expected_price:
                        slippage = abs(order.avg_fill_price - expected_price) / expected_price
                        slippages.append(slippage)
            
            avg_slippage = np.mean(slippages) if slippages else 0
        else:
            avg_slippage = 0
        
        # Execution speed
        execution_times = []
        for order in filled_orders:
            if order.fills:
                exec_time = (order.fills[-1]['timestamp'] - order.timestamp).total_seconds()
                execution_times.append(exec_time)
        
        avg_execution_time = np.mean(execution_times) if execution_times else 0
        
        return {
            'total_orders': total_orders,
            'fill_rate': fill_rate,
            'avg_slippage_bps': avg_slippage * 10000,  # In basis points
            'avg_execution_time_ms': avg_execution_time * 1000,
            'partial_fills': len([o for o in self.order_history if o.status == OrderStatus.PARTIAL]),
            'rejected_orders': len([o for o in self.order_history if o.status == OrderStatus.REJECTED])
        }


# Example usage
async def demo_market_simulator():
    """Demonstrate the market simulator"""
    simulator = MarketMicrostructureSimulator({
        'base_spread_bps': 2,
        'market_impact_factor': 0.1,
        'latency_ms': 50
    })
    
    # Simulate order book
    order_book = simulator.simulate_order_book(
        symbol='AAPL',
        mid_price=150.00,
        volume=1000000,
        volatility=0.02,
        timestamp=datetime.now()
    )
    
    print(f"Order Book for AAPL:")
    print(f"Best Bid: ${order_book.best_bid:.2f}")
    print(f"Best Ask: ${order_book.best_ask:.2f}")
    print(f"Spread: ${order_book.spread:.2f}")
    
    # Create and execute a market order
    order = Order(
        order_id='TEST001',
        symbol='AAPL',
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000
    )
    
    market_data = {
        'avg_daily_volume': 50000000,
        'volatility': 0.02
    }
    
    executed_order = await simulator.execute_order(order, order_book, market_data)
    
    print(f"\nOrder Execution:")
    print(f"Status: {executed_order.status}")
    print(f"Filled Quantity: {executed_order.filled_quantity}")
    print(f"Avg Fill Price: ${executed_order.avg_fill_price:.2f}")
    print(f"Number of Fills: {len(executed_order.fills)}")
    
    # Get execution analytics
    analytics = simulator.get_execution_analytics()
    print(f"\nExecution Analytics:")
    for key, value in analytics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(demo_market_simulator()) 