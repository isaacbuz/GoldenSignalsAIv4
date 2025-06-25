"""
Execution Management MCP Server
Provides trade execution capabilities via MCP protocol
"""

import sys
sys.path.append('..')

from base_server import MCPServer
from typing import Dict, Any, List, Optional
import asyncio
import uuid
from datetime import datetime
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class ExecutionMCPServer(MCPServer):
    """MCP server for execution management"""
    
    def __init__(self):
        super().__init__("Execution Management Server", 8503)
        self.capabilities = [
            "execution.submit_order",
            "execution.cancel_order",
            "execution.modify_order",
            "execution.get_order_status",
            "execution.get_executions",
            "execution.optimize_execution"
        ]
        self.orders = {}
        self.executions = []
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle execution requests"""
        method = request.get("method")
        params = request.get("params", {})
        
        try:
            if method == "execution.submit_order":
                return await self.submit_order(params)
            elif method == "execution.cancel_order":
                return await self.cancel_order(params)
            elif method == "execution.modify_order":
                return await self.modify_order(params)
            elif method == "execution.get_order_status":
                return await self.get_order_status(params)
            elif method == "execution.get_executions":
                return await self.get_executions(params)
            elif method == "execution.optimize_execution":
                return await self.optimize_execution(params)
            elif method == "capabilities":
                return {"capabilities": self.capabilities}
            else:
                return {"error": f"Unknown method: {method}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def submit_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Submit new order"""
        order_id = str(uuid.uuid4())
        
        order = {
            "order_id": order_id,
            "symbol": params.get("symbol"),
            "side": params.get("side"),  # buy/sell
            "quantity": params.get("quantity"),
            "order_type": params.get("order_type", "market"),
            "limit_price": params.get("limit_price"),
            "stop_price": params.get("stop_price"),
            "time_in_force": params.get("time_in_force", "DAY"),
            "status": OrderStatus.PENDING.value,
            "submitted_at": datetime.now().isoformat(),
            "filled_quantity": 0,
            "average_price": 0,
            "commission": 0
        }
        
        self.orders[order_id] = order
        
        # Simulate order processing
        asyncio.create_task(self._process_order(order_id))
        
        return {
            "order_id": order_id,
            "status": "submitted",
            "message": "Order submitted successfully"
        }
    
    async def _process_order(self, order_id: str):
        """Simulate order processing"""
        await asyncio.sleep(1)  # Simulate processing delay
        
        order = self.orders.get(order_id)
        if order and order["status"] == OrderStatus.PENDING.value:
            # Mock fill
            order["status"] = OrderStatus.FILLED.value
            order["filled_quantity"] = order["quantity"]
            order["average_price"] = order.get("limit_price", 100.0)
            order["filled_at"] = datetime.now().isoformat()
            
            # Create execution record
            execution = {
                "execution_id": str(uuid.uuid4()),
                "order_id": order_id,
                "symbol": order["symbol"],
                "side": order["side"],
                "quantity": order["quantity"],
                "price": order["average_price"],
                "commission": order["quantity"] * 0.001,  # $0.001 per share
                "executed_at": datetime.now().isoformat()
            }
            
            self.executions.append(execution)
            
            # Broadcast fill
            await self.broadcast({
                "event": "order_filled",
                "order_id": order_id,
                "execution": execution
            })
    
    async def cancel_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel existing order"""
        order_id = params.get("order_id")
        
        if order_id not in self.orders:
            return {"error": "Order not found"}
        
        order = self.orders[order_id]
        
        if order["status"] in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value]:
            return {"error": f"Cannot cancel {order['status']} order"}
        
        order["status"] = OrderStatus.CANCELLED.value
        order["cancelled_at"] = datetime.now().isoformat()
        
        return {
            "order_id": order_id,
            "status": "cancelled",
            "message": "Order cancelled successfully"
        }
    
    async def modify_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Modify existing order"""
        order_id = params.get("order_id")
        
        if order_id not in self.orders:
            return {"error": "Order not found"}
        
        order = self.orders[order_id]
        
        if order["status"] != OrderStatus.PENDING.value:
            return {"error": "Can only modify pending orders"}
        
        # Update allowed fields
        if "quantity" in params:
            order["quantity"] = params["quantity"]
        if "limit_price" in params:
            order["limit_price"] = params["limit_price"]
        if "stop_price" in params:
            order["stop_price"] = params["stop_price"]
            
        order["modified_at"] = datetime.now().isoformat()
        
        return {
            "order_id": order_id,
            "status": "modified",
            "message": "Order modified successfully"
        }
    
    async def get_order_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get order status"""
        order_id = params.get("order_id")
        
        if order_id not in self.orders:
            return {"error": "Order not found"}
        
        order = self.orders[order_id]
        
        return {
            "order": order,
            "can_cancel": order["status"] == OrderStatus.PENDING.value,
            "can_modify": order["status"] == OrderStatus.PENDING.value
        }
    
    async def get_executions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get execution history"""
        symbol = params.get("symbol")
        start_time = params.get("start_time")
        end_time = params.get("end_time")
        
        filtered_executions = self.executions
        
        if symbol:
            filtered_executions = [e for e in filtered_executions if e["symbol"] == symbol]
            
        return {
            "executions": filtered_executions[-100:],  # Last 100
            "total_count": len(filtered_executions)
        }
    
    async def optimize_execution(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize order execution"""
        symbol = params.get("symbol")
        quantity = params.get("quantity")
        side = params.get("side")
        urgency = params.get("urgency", "normal")  # normal, high, low
        
        # Mock execution optimization
        recommendations = {
            "execution_strategy": "TWAP" if urgency == "low" else "VWAP",
            "slice_count": max(1, quantity // 1000),
            "timing": {
                "start_time": "09:45",
                "end_time": "15:45",
                "avoid_times": ["09:30-09:45", "15:45-16:00"]
            },
            "price_limits": {
                "max_spread": 0.02,
                "price_improvement": 0.001
            },
            "venues": [
                {"name": "PRIMARY", "percentage": 60},
                {"name": "DARK_POOL", "percentage": 30},
                {"name": "ECN", "percentage": 10}
            ]
        }
        
        if urgency == "high":
            recommendations["execution_strategy"] = "AGGRESSIVE"
            recommendations["slice_count"] = 1
            
        return {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "recommendations": recommendations,
            "estimated_cost": quantity * 0.001,  # Commission estimate
            "estimated_market_impact": 0.0005  # 5 bps
        }

async def main():
    """Run Execution Management MCP Server"""
    server = ExecutionMCPServer()
    await server.start()
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
