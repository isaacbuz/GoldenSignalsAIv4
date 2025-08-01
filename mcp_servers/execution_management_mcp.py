"""
Execution Management MCP Server
Handles order routing, execution strategies, and trade monitoring
Issue #194: MCP-5: Build Execution Management MCP Server
"""

from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Tuple, Callable
import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict, field
import numpy as np
from collections import defaultdict, deque
import time
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"  # Percentage of Volume


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionAlgo(Enum):
    """Execution algorithms"""
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    ADAPTIVE = "adaptive"
    SMART = "smart"
    DARK_POOL = "dark_pool"
    LIQUIDITY_SEEKING = "liquidity_seeking"


class Venue(Enum):
    """Trading venues"""
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    ARCA = "arca"
    BATS = "bats"
    IEX = "iex"
    DARK_POOL = "dark_pool"
    ALL = "all"


@dataclass
class Order:
    """Order structure"""
    id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    algo: ExecutionAlgo = ExecutionAlgo.SMART
    venue_preference: List[Venue] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'algo': self.algo.value,
            'venue_preference': [v.value for v in self.venue_preference],
            'created_at': self.created_at.isoformat()
        }


@dataclass
class Execution:
    """Execution/Fill structure"""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    venue: Venue
    timestamp: datetime
    fees: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'side': self.side.value,
            'venue': self.venue.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class OrderState:
    """Complete order state"""
    order: Order
    status: OrderStatus
    filled_quantity: int = 0
    average_price: float = 0.0
    executions: List[Execution] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None

    @property
    def remaining_quantity(self) -> int:
        return self.order.quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.order.quantity if self.order.quantity > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order': self.order.to_dict(),
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_price': self.average_price,
            'fill_rate': self.fill_rate,
            'execution_count': len(self.executions),
            'last_update': self.last_update.isoformat(),
            'error_message': self.error_message
        }


class MarketMicrostructure:
    """Simulates market microstructure for realistic execution"""

    def __init__(self):
        self.bid_ask_spreads = defaultdict(lambda: 0.01)  # Default 1 cent spread
        self.market_impact_factors = defaultdict(lambda: 0.0001)  # 1 bp impact
        self.liquidity_profiles = defaultdict(lambda: "normal")

    def get_execution_price(self, symbol: str, side: OrderSide, quantity: int,
                          venue: Venue, base_price: float) -> Tuple[float, float]:
        """Calculate execution price with market impact"""
        spread = self.bid_ask_spreads[symbol]
        impact_factor = self.market_impact_factors[symbol]

        # Base spread cost
        if side == OrderSide.BUY:
            price = base_price + spread / 2
        else:
            price = base_price - spread / 2

        # Market impact
        impact = impact_factor * np.sqrt(quantity)
        if side == OrderSide.BUY:
            price += base_price * impact
        else:
            price -= base_price * impact

        # Venue-specific adjustments
        if venue == Venue.DARK_POOL:
            # Dark pools often offer mid-point execution
            price = base_price
        elif venue == Venue.IEX:
            # IEX has speed bump, might get slightly better price
            price *= 0.9999

        # Calculate fees (simplified)
        fees = quantity * 0.00005  # $0.05 per 1000 shares

        return price, fees

    def estimate_fill_probability(self, order: Order, market_conditions: Dict[str, Any]) -> float:
        """Estimate probability of order being filled"""
        if order.order_type == OrderType.MARKET:
            return 1.0

        # For limit orders, depends on price relative to market
        if order.order_type == OrderType.LIMIT and order.price:
            current_price = market_conditions.get('price', 100)

            if order.side == OrderSide.BUY:
                # Buy limit below market has lower fill probability
                if order.price < current_price:
                    return max(0.1, 1 - (current_price - order.price) / current_price)
            else:
                # Sell limit above market has lower fill probability
                if order.price > current_price:
                    return max(0.1, 1 - (order.price - current_price) / current_price)

        return 0.5


class SmartOrderRouter:
    """Smart order routing logic"""

    def __init__(self, microstructure: MarketMicrostructure):
        self.microstructure = microstructure
        self.venue_latencies = {
            Venue.NYSE: 1,
            Venue.NASDAQ: 1,
            Venue.ARCA: 2,
            Venue.BATS: 2,
            Venue.IEX: 350,  # IEX speed bump
            Venue.DARK_POOL: 5
        }

    async def route_order(self, order: Order, market_data: Dict[str, Any]) -> List[Tuple[Venue, int]]:
        """Determine optimal routing for order"""
        if order.venue_preference and order.venue_preference[0] != Venue.ALL:
            # Respect explicit venue preference
            return [(order.venue_preference[0], order.quantity)]

        # Smart routing based on order characteristics
        if order.quantity > 10000:
            # Large orders: use multiple venues and dark pools
            return self._route_large_order(order)
        elif order.algo == ExecutionAlgo.DARK_POOL:
            # Dark pool preference
            return [(Venue.DARK_POOL, order.quantity)]
        elif order.algo == ExecutionAlgo.AGGRESSIVE:
            # Fast execution on primary venues
            return self._route_aggressive(order)
        else:
            # Default smart routing
            return self._route_smart(order)

    def _route_large_order(self, order: Order) -> List[Tuple[Venue, int]]:
        """Route large orders across multiple venues"""
        routes = []
        remaining = order.quantity

        # 40% to dark pool
        dark_qty = int(order.quantity * 0.4)
        routes.append((Venue.DARK_POOL, dark_qty))
        remaining -= dark_qty

        # Split remainder across lit venues
        venues = [Venue.NYSE, Venue.NASDAQ, Venue.ARCA]
        for venue in venues:
            qty = remaining // len(venues)
            routes.append((venue, qty))

        return routes

    def _route_aggressive(self, order: Order) -> List[Tuple[Venue, int]]:
        """Route for aggressive execution"""
        # Use fastest venues
        fast_venues = sorted(
            [Venue.NYSE, Venue.NASDAQ, Venue.BATS],
            key=lambda v: self.venue_latencies[v]
        )

        # Send to fastest venue
        return [(fast_venues[0], order.quantity)]

    def _route_smart(self, order: Order) -> List[Tuple[Venue, int]]:
        """Default smart routing"""
        # Simple split across major venues
        venues = [Venue.NYSE, Venue.NASDAQ, Venue.ARCA]
        qty_per_venue = order.quantity // len(venues)

        routes = [(venue, qty_per_venue) for venue in venues]

        # Add remainder to first venue
        remainder = order.quantity - (qty_per_venue * len(venues))
        if remainder > 0:
            routes[0] = (routes[0][0], routes[0][1] + remainder)

        return routes


class ExecutionManagementMCP:
    """
    MCP Server for order execution management
    Handles order routing, execution algorithms, and trade monitoring
    """

    def __init__(self):
        self.app = FastAPI(title="Execution Management MCP Server")

        # Order management
        self.orders: Dict[str, OrderState] = {}
        self.active_orders: Dict[str, OrderState] = {}

        # Market microstructure simulation
        self.microstructure = MarketMicrostructure()
        self.smart_router = SmartOrderRouter(self.microstructure)

        # Execution algorithms
        self.execution_algos: Dict[ExecutionAlgo, Callable] = {
            ExecutionAlgo.TWAP: self._execute_twap,
            ExecutionAlgo.VWAP: self._execute_vwap,
            ExecutionAlgo.POV: self._execute_pov,
            ExecutionAlgo.ADAPTIVE: self._execute_adaptive
        }

        # WebSocket clients
        self.websocket_clients: List[WebSocket] = []

        # Metrics
        self.metrics = {
            'total_orders': 0,
            'total_executions': 0,
            'total_volume': 0,
            'average_slippage': 0,
            'fill_rate': 0
        }

        self._setup_routes()

        # Start background tasks
        asyncio.create_task(self._execution_engine())
        asyncio.create_task(self._order_monitoring())

    def _setup_routes(self):
        """Set up FastAPI routes"""

        @self.app.get("/")
        async def root():
            return {
                "service": "Execution Management MCP",
                "status": "active",
                "active_orders": len(self.active_orders),
                "metrics": self.metrics
            }

        @self.app.get("/tools")
        async def list_tools():
            """List available execution tools"""
            return {
                "tools": [
                    {
                        "name": "submit_order",
                        "description": "Submit a new order for execution",
                        "parameters": {
                            "symbol": "string",
                            "side": "string (buy/sell)",
                            "quantity": "integer",
                            "order_type": "string (market/limit/stop/etc)",
                            "price": "number (required for limit orders)",
                            "algo": "string (smart/aggressive/passive/adaptive/twap/vwap)",
                            "time_in_force": "string (DAY/GTC/IOC/FOK)"
                        }
                    },
                    {
                        "name": "cancel_order",
                        "description": "Cancel an active order",
                        "parameters": {
                            "order_id": "string"
                        }
                    },
                    {
                        "name": "modify_order",
                        "description": "Modify an active order",
                        "parameters": {
                            "order_id": "string",
                            "new_quantity": "integer (optional)",
                            "new_price": "number (optional)"
                        }
                    },
                    {
                        "name": "get_order_status",
                        "description": "Get detailed order status",
                        "parameters": {
                            "order_id": "string"
                        }
                    },
                    {
                        "name": "get_executions",
                        "description": "Get execution history",
                        "parameters": {
                            "order_id": "string (optional)",
                            "symbol": "string (optional)",
                            "start_time": "string (ISO format)",
                            "end_time": "string (ISO format)"
                        }
                    },
                    {
                        "name": "estimate_impact",
                        "description": "Estimate market impact for an order",
                        "parameters": {
                            "symbol": "string",
                            "side": "string",
                            "quantity": "integer"
                        }
                    }
                ]
            }

        @self.app.post("/call")
        async def call_tool(request: Dict[str, Any], background_tasks: BackgroundTasks):
            """Execute an execution management tool"""
            tool_name = request.get("tool")
            params = request.get("parameters", {})

            try:
                if tool_name == "submit_order":
                    return await self._submit_order(params, background_tasks)
                elif tool_name == "cancel_order":
                    return await self._cancel_order(params)
                elif tool_name == "modify_order":
                    return await self._modify_order(params)
                elif tool_name == "get_order_status":
                    return await self._get_order_status(params)
                elif tool_name == "get_executions":
                    return await self._get_executions(params)
                elif tool_name == "estimate_impact":
                    return await self._estimate_impact(params)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

            except Exception as e:
                logger.error(f"Error in tool call {tool_name}: {e}")
                return {"error": str(e), "tool": tool_name}

        @self.app.get("/orders")
        async def list_orders(
            status: Optional[str] = None,
            symbol: Optional[str] = None,
            limit: int = 100
        ):
            """List orders with optional filters"""
            orders = list(self.orders.values())

            if status:
                order_status = OrderStatus(status)
                orders = [o for o in orders if o.status == order_status]

            if symbol:
                orders = [o for o in orders if o.order.symbol == symbol]

            # Sort by last update descending
            orders.sort(key=lambda x: x.last_update, reverse=True)

            return {
                "orders": [o.to_dict() for o in orders[:limit]],
                "total": len(orders)
            }

        @self.app.websocket("/ws/executions")
        async def websocket_executions(websocket: WebSocket):
            """WebSocket for real-time execution updates"""
            await websocket.accept()
            self.websocket_clients.append(websocket)

            try:
                while True:
                    # Keep connection alive
                    await asyncio.sleep(30)
                    await websocket.send_json({"type": "ping"})

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_clients.remove(websocket)
                await websocket.close()

    async def _submit_order(self, params: Dict[str, Any],
                          background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """Submit a new order"""
        # Create order
        order = Order(
            id=str(uuid.uuid4()),
            symbol=params['symbol'],
            side=OrderSide(params['side']),
            quantity=params['quantity'],
            order_type=OrderType(params.get('order_type', 'market')),
            price=params.get('price'),
            stop_price=params.get('stop_price'),
            time_in_force=params.get('time_in_force', 'DAY'),
            algo=ExecutionAlgo(params.get('algo', 'smart'))
        )

        # Validate order
        validation_error = self._validate_order(order)
        if validation_error:
            return {"error": validation_error, "order_id": order.id}

        # Create order state
        order_state = OrderState(
            order=order,
            status=OrderStatus.PENDING
        )

        # Store order
        self.orders[order.id] = order_state
        self.active_orders[order.id] = order_state

        # Update metrics
        self.metrics['total_orders'] += 1

        # Start execution in background
        background_tasks.add_task(self._process_order, order_state)

        # Send immediate confirmation
        await self._broadcast_order_update(order_state)

        return {
            "order_id": order.id,
            "status": "submitted",
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value
        }

    async def _cancel_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel an active order"""
        order_id = params.get('order_id')

        if order_id not in self.active_orders:
            return {"error": f"Order {order_id} not found or not active"}

        order_state = self.active_orders[order_id]

        # Check if cancellable
        if order_state.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return {"error": f"Order {order_id} is already {order_state.status.value}"}

        # Cancel order
        order_state.status = OrderStatus.CANCELLED
        order_state.last_update = datetime.now()

        # Remove from active orders
        del self.active_orders[order_id]

        # Broadcast update
        await self._broadcast_order_update(order_state)

        return {
            "order_id": order_id,
            "status": "cancelled",
            "filled_quantity": order_state.filled_quantity,
            "remaining_quantity": order_state.remaining_quantity
        }

    async def _modify_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an active order"""
        order_id = params.get('order_id')

        if order_id not in self.active_orders:
            return {"error": f"Order {order_id} not found or not active"}

        order_state = self.active_orders[order_id]

        # Check if modifiable
        if order_state.status != OrderStatus.SUBMITTED:
            return {"error": f"Order {order_id} cannot be modified in status {order_state.status.value}"}

        # Apply modifications
        modified = False

        if 'new_quantity' in params:
            new_qty = params['new_quantity']
            if new_qty > order_state.filled_quantity:
                order_state.order.quantity = new_qty
                modified = True

        if 'new_price' in params and order_state.order.order_type == OrderType.LIMIT:
            order_state.order.price = params['new_price']
            modified = True

        if modified:
            order_state.last_update = datetime.now()
            await self._broadcast_order_update(order_state)

            return {
                "order_id": order_id,
                "status": "modified",
                "new_quantity": order_state.order.quantity,
                "new_price": order_state.order.price
            }
        else:
            return {"error": "No valid modifications provided"}

    async def _get_order_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed order status"""
        order_id = params.get('order_id')

        if order_id not in self.orders:
            return {"error": f"Order {order_id} not found"}

        order_state = self.orders[order_id]

        return {
            "order": order_state.to_dict(),
            "executions": [e.to_dict() for e in order_state.executions],
            "slippage": self._calculate_slippage(order_state),
            "execution_quality": self._assess_execution_quality(order_state)
        }

    async def _get_executions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get execution history"""
        executions = []

        # Filter by parameters
        for order_state in self.orders.values():
            for execution in order_state.executions:
                # Apply filters
                if params.get('order_id') and execution.order_id != params['order_id']:
                    continue
                if params.get('symbol') and execution.symbol != params['symbol']:
                    continue

                executions.append(execution.to_dict())

        # Sort by timestamp descending
        executions.sort(key=lambda x: x['timestamp'], reverse=True)

        return {
            "executions": executions,
            "total": len(executions),
            "total_volume": sum(e['quantity'] for e in executions),
            "total_value": sum(e['quantity'] * e['price'] for e in executions)
        }

    async def _estimate_impact(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate market impact for an order"""
        symbol = params.get('symbol')
        side = OrderSide(params.get('side'))
        quantity = params.get('quantity')

        # Get current market conditions
        base_price = 100.0  # In production, would get real price

        # Calculate impact
        impact_factor = self.microstructure.market_impact_factors[symbol]
        spread = self.microstructure.bid_ask_spreads[symbol]

        # Linear + square root impact model
        linear_impact = impact_factor * quantity / 10000
        sqrt_impact = impact_factor * np.sqrt(quantity)
        total_impact = linear_impact + sqrt_impact

        # Spread cost
        spread_cost = spread / 2

        # Total cost
        if side == OrderSide.BUY:
            estimated_price = base_price * (1 + total_impact) + spread_cost
        else:
            estimated_price = base_price * (1 - total_impact) - spread_cost

        slippage = abs(estimated_price - base_price)
        slippage_pct = (slippage / base_price) * 100

        return {
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "base_price": base_price,
            "estimated_price": estimated_price,
            "spread_cost": spread_cost,
            "market_impact": total_impact * base_price,
            "total_slippage": slippage,
            "slippage_pct": slippage_pct,
            "estimated_cost": quantity * estimated_price,
            "recommendations": self._get_execution_recommendations(quantity, slippage_pct)
        }

    def _validate_order(self, order: Order) -> Optional[str]:
        """Validate order parameters"""
        if order.quantity <= 0:
            return "Quantity must be positive"

        if order.order_type == OrderType.LIMIT and order.price is None:
            return "Limit orders require a price"

        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            return "Stop orders require a stop price"

        return None

    async def _process_order(self, order_state: OrderState):
        """Process order execution"""
        try:
            # Update status
            order_state.status = OrderStatus.SUBMITTED
            order_state.last_update = datetime.now()
            await self._broadcast_order_update(order_state)

            # Route order
            order = order_state.order
            market_data = await self._get_market_data(order.symbol)
            routes = await self.smart_router.route_order(order, market_data)

            # Execute based on algorithm
            if order.algo in [ExecutionAlgo.TWAP, ExecutionAlgo.VWAP, ExecutionAlgo.POV]:
                # Algorithmic execution
                await self.execution_algos[order.algo](order_state, routes)
            else:
                # Direct execution
                await self._execute_direct(order_state, routes)

        except Exception as e:
            logger.error(f"Error processing order {order_state.order.id}: {e}")
            order_state.status = OrderStatus.REJECTED
            order_state.error_message = str(e)
            order_state.last_update = datetime.now()
            await self._broadcast_order_update(order_state)

        finally:
            # Remove from active orders if terminal state
            if order_state.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                if order_state.order.id in self.active_orders:
                    del self.active_orders[order_state.order.id]

    async def _execute_direct(self, order_state: OrderState, routes: List[Tuple[Venue, int]]):
        """Execute order directly on specified routes"""
        order = order_state.order

        for venue, quantity in routes:
            if order_state.remaining_quantity <= 0:
                break

            # Simulate execution
            exec_qty = min(quantity, order_state.remaining_quantity)

            # Get execution price
            market_price = await self._get_market_price(order.symbol)
            exec_price, fees = self.microstructure.get_execution_price(
                order.symbol, order.side, exec_qty, venue, market_price
            )

            # Check limit price
            if order.order_type == OrderType.LIMIT and order.price:
                if order.side == OrderSide.BUY and exec_price > order.price:
                    continue  # Skip this execution
                elif order.side == OrderSide.SELL and exec_price < order.price:
                    continue  # Skip this execution

            # Create execution
            execution = Execution(
                id=str(uuid.uuid4()),
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=exec_qty,
                price=exec_price,
                venue=venue,
                timestamp=datetime.now(),
                fees=fees
            )

            # Update order state
            await self._apply_execution(order_state, execution)

            # Small delay between executions
            await asyncio.sleep(0.1)

    async def _execute_twap(self, order_state: OrderState, routes: List[Tuple[Venue, int]]):
        """Execute order using Time-Weighted Average Price algorithm"""
        order = order_state.order

        # Split order into time slices
        slices = 10  # Execute over 10 intervals
        slice_quantity = order.quantity // slices

        for i in range(slices):
            if order_state.remaining_quantity <= 0:
                break

            # Execute slice
            slice_routes = [(v, slice_quantity * q // order.quantity) for v, q in routes]
            await self._execute_direct(order_state, slice_routes)

            # Wait between slices (in production, would be minutes)
            await asyncio.sleep(1)

    async def _execute_vwap(self, order_state: OrderState, routes: List[Tuple[Venue, int]]):
        """Execute order using Volume-Weighted Average Price algorithm"""
        # Simplified VWAP - in production would use historical volume curves
        # For now, similar to TWAP but with varying slice sizes
        order = order_state.order

        # Volume distribution (U-shaped for typical trading day)
        volume_weights = [0.15, 0.10, 0.08, 0.07, 0.10, 0.10, 0.07, 0.08, 0.10, 0.15]

        for weight in volume_weights:
            if order_state.remaining_quantity <= 0:
                break

            # Execute weighted slice
            slice_quantity = int(order.quantity * weight)
            slice_routes = [(v, slice_quantity * q // order.quantity) for v, q in routes]
            await self._execute_direct(order_state, slice_routes)

            await asyncio.sleep(1)

    async def _execute_pov(self, order_state: OrderState, routes: List[Tuple[Venue, int]]):
        """Execute order as Percentage of Volume"""
        # Simplified POV - execute as 10% of market volume
        target_pov = 0.10

        # In production, would monitor real market volume
        # For demo, execute in small chunks
        chunk_size = max(100, order_state.order.quantity // 20)

        while order_state.remaining_quantity > 0:
            exec_qty = min(chunk_size, order_state.remaining_quantity)
            chunk_routes = [(routes[0][0], exec_qty)]  # Use primary route

            await self._execute_direct(order_state, chunk_routes)
            await asyncio.sleep(0.5)

    async def _execute_adaptive(self, order_state: OrderState, routes: List[Tuple[Venue, int]]):
        """Adaptive execution based on market conditions"""
        # Monitor market conditions and adapt execution
        order = order_state.order

        while order_state.remaining_quantity > 0:
            # Get current market conditions
            volatility = np.random.uniform(0.1, 0.3)  # Simulated

            # Adapt execution size based on volatility
            if volatility > 0.25:
                # High volatility - execute smaller chunks
                chunk_size = max(100, order.quantity // 50)
            else:
                # Low volatility - execute larger chunks
                chunk_size = max(500, order.quantity // 10)

            exec_qty = min(chunk_size, order_state.remaining_quantity)
            chunk_routes = [(routes[0][0], exec_qty)]

            await self._execute_direct(order_state, chunk_routes)
            await asyncio.sleep(0.2)

    async def _apply_execution(self, order_state: OrderState, execution: Execution):
        """Apply execution to order state"""
        # Update filled quantity
        order_state.filled_quantity += execution.quantity

        # Update average price
        if order_state.filled_quantity > 0:
            total_value = (order_state.average_price * (order_state.filled_quantity - execution.quantity) +
                          execution.price * execution.quantity)
            order_state.average_price = total_value / order_state.filled_quantity

        # Add execution
        order_state.executions.append(execution)

        # Update status
        if order_state.filled_quantity >= order_state.order.quantity:
            order_state.status = OrderStatus.FILLED
        else:
            order_state.status = OrderStatus.PARTIAL

        order_state.last_update = datetime.now()

        # Update metrics
        self.metrics['total_executions'] += 1
        self.metrics['total_volume'] += execution.quantity

        # Broadcast updates
        await self._broadcast_execution(execution)
        await self._broadcast_order_update(order_state)

    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        # Simulated market data
        return {
            'symbol': symbol,
            'price': 100.0 + np.random.randn() * 2,
            'bid': 99.95,
            'ask': 100.05,
            'volume': 1000000,
            'volatility': 0.20
        }

    async def _get_market_price(self, symbol: str) -> float:
        """Get current market price"""
        data = await self._get_market_data(symbol)
        return data['price']

    def _calculate_slippage(self, order_state: OrderState) -> Dict[str, Any]:
        """Calculate execution slippage"""
        if not order_state.executions:
            return {"slippage": 0, "slippage_pct": 0}

        # Expected price (first execution price or limit price)
        if order_state.order.order_type == OrderType.LIMIT and order_state.order.price:
            expected_price = order_state.order.price
        else:
            expected_price = order_state.executions[0].price

        # Actual average price
        actual_price = order_state.average_price

        # Calculate slippage
        if order_state.order.side == OrderSide.BUY:
            slippage = actual_price - expected_price
        else:
            slippage = expected_price - actual_price

        slippage_pct = (slippage / expected_price) * 100 if expected_price > 0 else 0

        return {
            "expected_price": expected_price,
            "actual_price": actual_price,
            "slippage": slippage,
            "slippage_pct": slippage_pct,
            "slippage_cost": slippage * order_state.filled_quantity
        }

    def _assess_execution_quality(self, order_state: OrderState) -> Dict[str, Any]:
        """Assess overall execution quality"""
        if not order_state.executions:
            return {"quality_score": 0, "assessment": "No executions"}

        # Calculate quality metrics
        slippage_data = self._calculate_slippage(order_state)
        fill_rate = order_state.fill_rate
        execution_time = (order_state.last_update - order_state.order.created_at).total_seconds()

        # Quality score (0-100)
        quality_score = 100

        # Penalize for slippage
        quality_score -= min(abs(slippage_data['slippage_pct']) * 10, 30)

        # Penalize for partial fills
        quality_score -= (1 - fill_rate) * 20

        # Penalize for slow execution
        if execution_time > 60:  # More than 1 minute
            quality_score -= min(execution_time / 60, 20)

        quality_score = max(0, quality_score)

        # Assessment
        if quality_score >= 90:
            assessment = "Excellent"
        elif quality_score >= 75:
            assessment = "Good"
        elif quality_score >= 60:
            assessment = "Fair"
        else:
            assessment = "Poor"

        return {
            "quality_score": quality_score,
            "assessment": assessment,
            "fill_rate": fill_rate,
            "execution_time_seconds": execution_time,
            "venue_distribution": self._get_venue_distribution(order_state)
        }

    def _get_venue_distribution(self, order_state: OrderState) -> Dict[str, float]:
        """Get distribution of executions across venues"""
        venue_quantities = defaultdict(int)

        for execution in order_state.executions:
            venue_quantities[execution.venue.value] += execution.quantity

        total_executed = order_state.filled_quantity

        return {
            venue: (qty / total_executed) * 100
            for venue, qty in venue_quantities.items()
        } if total_executed > 0 else {}

    def _get_execution_recommendations(self, quantity: int, slippage_pct: float) -> List[str]:
        """Get recommendations for order execution"""
        recommendations = []

        if quantity > 50000:
            recommendations.append("Consider using VWAP or TWAP algorithm for large order")

        if slippage_pct > 0.5:
            recommendations.append("High expected slippage - consider dark pool execution")

        if quantity > 100000:
            recommendations.append("Split order across multiple venues to minimize impact")

        return recommendations

    async def _broadcast_order_update(self, order_state: OrderState):
        """Broadcast order update to WebSocket clients"""
        update = {
            "type": "order_update",
            "order": order_state.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

        for client in self.websocket_clients:
            try:
                await client.send_json(update)
            except Exception as e:
                logger.error(f"Failed to send order update: {e}")

    async def _broadcast_execution(self, execution: Execution):
        """Broadcast execution to WebSocket clients"""
        update = {
            "type": "execution",
            "execution": execution.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

        for client in self.websocket_clients:
            try:
                await client.send_json(update)
            except Exception as e:
                logger.error(f"Failed to send execution update: {e}")

    async def _execution_engine(self):
        """Background execution engine"""
        # This would connect to real brokers/exchanges in production
        while True:
            try:
                # Process any pending orders
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Execution engine error: {e}")

    async def _order_monitoring(self):
        """Monitor orders for timeouts and other conditions"""
        while True:
            try:
                now = datetime.now()

                for order_id, order_state in list(self.active_orders.items()):
                    # Check for day order expiration
                    if order_state.order.time_in_force == "DAY":
                        order_age = (now - order_state.order.created_at).total_seconds()
                        if order_age > 23400:  # 6.5 hours (market hours)
                            order_state.status = OrderStatus.EXPIRED
                            order_state.last_update = now
                            del self.active_orders[order_id]
                            await self._broadcast_order_update(order_state)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Order monitoring error: {e}")


# Demo function
async def demo_execution_management_mcp():
    """Demonstrate Execution Management MCP functionality"""
    import uvicorn

    logger.info("Starting Execution Management MCP Server demo...")

    # Create server
    server = ExecutionManagementMCP()

    # Run server
    config = uvicorn.Config(
        app=server.app,
        host="0.0.0.0",
        port=8194,
        log_level="info"
    )

    server_instance = uvicorn.Server(config)
    await server_instance.serve()


if __name__ == "__main__":
    asyncio.run(demo_execution_management_mcp())
