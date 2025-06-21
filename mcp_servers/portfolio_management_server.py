#!/usr/bin/env python3
"""
MCP Portfolio Management Server for GoldenSignalsAI
Week 4 Implementation: Portfolio tracking, risk management, and position sizing
"""
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict
import random

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_date: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def pnl_percentage(self) -> float:
        if self.entry_price == 0:
            return 0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

@dataclass
class Portfolio:
    """Portfolio state and metrics"""
    cash: float
    positions: Dict[str, Position]
    initial_capital: float
    
    @property
    def total_market_value(self) -> float:
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_value(self) -> float:
        return self.cash + self.total_market_value
    
    @property
    def total_return(self) -> float:
        if self.initial_capital == 0:
            return 0
        return ((self.total_value - self.initial_capital) / self.initial_capital) * 100
    
    @property
    def position_count(self) -> int:
        return len(self.positions)

class PortfolioManagementServer:
    """Portfolio management server with risk management capabilities"""
    
    def __init__(self):
        self.server = Server("goldensignals-portfolio")
        
        # Portfolio state
        self.portfolio = Portfolio(
            cash=100000.0,  # Starting with $100k
            positions={},
            initial_capital=100000.0
        )
        
        # Risk parameters
        self.risk_params = {
            "max_position_size": 0.20,  # Max 20% per position
            "max_risk_per_trade": 0.02,  # Max 2% risk per trade
            "max_portfolio_risk": 0.06,  # Max 6% portfolio risk
            "volatility_window": 20,
            "correlation_threshold": 0.7
        }
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        
        # Setup handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup tool and resource handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="get_portfolio_status",
                    description="Get current portfolio status including positions, P&L, and metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                types.Tool(
                    name="calculate_position_size",
                    description="Calculate optimal position size based on risk parameters and signal confidence",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Stock symbol"},
                            "price": {"type": "number", "description": "Current price"},
                            "signal_confidence": {"type": "number", "description": "Signal confidence (0-1)"},
                            "volatility": {"type": "number", "description": "Current volatility", "default": 0.02}
                        },
                        "required": ["symbol", "price", "signal_confidence"]
                    }
                ),
                types.Tool(
                    name="execute_trade",
                    description="Execute a trade (buy/sell) with risk management",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Stock symbol"},
                            "action": {"type": "string", "enum": ["buy", "sell"], "description": "Trade action"},
                            "quantity": {"type": "number", "description": "Number of shares"},
                            "price": {"type": "number", "description": "Execution price"},
                            "stop_loss": {"type": "number", "description": "Stop loss price", "default": None},
                            "take_profit": {"type": "number", "description": "Take profit price", "default": None}
                        },
                        "required": ["symbol", "action", "quantity", "price"]
                    }
                ),
                types.Tool(
                    name="get_risk_metrics",
                    description="Get current portfolio risk metrics and exposure analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                types.Tool(
                    name="rebalance_portfolio",
                    description="Get rebalancing recommendations based on target allocations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "target_allocations": {
                                "type": "object",
                                "description": "Target allocations by symbol (percentages)",
                                "additionalProperties": {"type": "number"}
                            }
                        },
                        "required": ["target_allocations"]
                    }
                ),
                types.Tool(
                    name="get_performance_analytics",
                    description="Get detailed performance analytics including Sharpe ratio, max drawdown, etc.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "period_days": {"type": "integer", "description": "Analysis period in days", "default": 30}
                        },
                        "required": []
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, 
            arguments: Optional[Dict[str, Any]] = None
        ) -> List[types.TextContent]:
            try:
                if name == "get_portfolio_status":
                    result = await self._get_portfolio_status()
                elif name == "calculate_position_size":
                    result = await self._calculate_position_size(arguments)
                elif name == "execute_trade":
                    result = await self._execute_trade(arguments)
                elif name == "get_risk_metrics":
                    result = await self._get_risk_metrics()
                elif name == "rebalance_portfolio":
                    result = await self._rebalance_portfolio(arguments)
                elif name == "get_performance_analytics":
                    result = await self._get_performance_analytics(arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[types.Resource]:
            return [
                types.Resource(
                    uri="portfolio://status/live",
                    name="Live Portfolio Status",
                    description="Real-time portfolio status and positions",
                    mimeType="application/json"
                ),
                types.Resource(
                    uri="portfolio://risk/metrics",
                    name="Risk Metrics",
                    description="Current portfolio risk metrics",
                    mimeType="application/json"
                ),
                types.Resource(
                    uri="portfolio://performance/history",
                    name="Performance History",
                    description="Historical portfolio performance",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "portfolio://status/live":
                status = await self._get_portfolio_status()
                return json.dumps(status, indent=2, default=str)
            elif uri == "portfolio://risk/metrics":
                metrics = await self._get_risk_metrics()
                return json.dumps(metrics, indent=2, default=str)
            elif uri == "portfolio://performance/history":
                history = {
                    "portfolio_history": self.portfolio_history[-100:],  # Last 100 snapshots
                    "trade_history": self.trade_history[-50:]  # Last 50 trades
                }
                return json.dumps(history, indent=2, default=str)
            else:
                return json.dumps({"error": f"Unknown resource: {uri}"})
    
    async def _get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        # Update current prices (simulate with random walk)
        for position in self.portfolio.positions.values():
            position.current_price *= (1 + random.uniform(-0.02, 0.02))
        
        positions_data = []
        for symbol, position in self.portfolio.positions.items():
            positions_data.append({
                "symbol": symbol,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "market_value": position.market_value,
                "unrealized_pnl": position.unrealized_pnl,
                "pnl_percentage": position.pnl_percentage,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "entry_date": position.entry_date.isoformat()
            })
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "cash": self.portfolio.cash,
            "total_market_value": self.portfolio.total_market_value,
            "total_value": self.portfolio.total_value,
            "initial_capital": self.portfolio.initial_capital,
            "total_return": self.portfolio.total_return,
            "position_count": self.portfolio.position_count,
            "positions": positions_data,
            "allocation": self._calculate_allocation()
        }
        
        # Record snapshot
        self.portfolio_history.append(status)
        
        return status
    
    async def _calculate_position_size(self, args: Dict) -> Dict:
        """Calculate optimal position size using Kelly Criterion"""
        symbol = args["symbol"]
        price = args["price"]
        confidence = args["signal_confidence"]
        volatility = args.get("volatility", 0.02)
        
        # Kelly Criterion calculation
        win_probability = confidence
        win_loss_ratio = 2.0  # Assume 2:1 reward/risk ratio
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        
        # Apply safety factor (use 25% of Kelly)
        safe_fraction = kelly_fraction * 0.25
        
        # Adjust for volatility
        vol_factor = 1.0 / (1.0 + volatility * 10)
        
        # Calculate position value
        max_position_value = self.portfolio.total_value * self.risk_params["max_position_size"]
        position_value = self.portfolio.total_value * safe_fraction * vol_factor
        position_value = min(position_value, max_position_value)
        
        # Calculate shares
        shares = int(position_value / price)
        
        # Calculate stop loss based on volatility
        stop_loss = price * (1 - volatility * 2)
        take_profit = price * (1 + volatility * 4)
        
        # Risk per share
        risk_per_share = price - stop_loss
        max_risk_amount = self.portfolio.total_value * self.risk_params["max_risk_per_trade"]
        max_shares_by_risk = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else shares
        
        final_shares = min(shares, max_shares_by_risk)
        
        return {
            "symbol": symbol,
            "recommended_shares": final_shares,
            "position_value": final_shares * price,
            "position_percentage": (final_shares * price) / self.portfolio.total_value * 100,
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_amount": final_shares * risk_per_share,
            "risk_percentage": (final_shares * risk_per_share) / self.portfolio.total_value * 100,
            "kelly_fraction": kelly_fraction,
            "adjusted_fraction": safe_fraction * vol_factor,
            "volatility_factor": vol_factor
        }
    
    async def _execute_trade(self, args: Dict) -> Dict:
        """Execute a trade with risk management"""
        symbol = args["symbol"]
        action = args["action"]
        quantity = args["quantity"]
        price = args["price"]
        stop_loss = args.get("stop_loss")
        take_profit = args.get("take_profit")
        
        timestamp = datetime.now()
        
        if action == "buy":
            # Check if we have enough cash
            required_cash = quantity * price
            if required_cash > self.portfolio.cash:
                return {
                    "status": "rejected",
                    "reason": "Insufficient cash",
                    "required": required_cash,
                    "available": self.portfolio.cash
                }
            
            # Check position size limit
            position_value = quantity * price
            if position_value > self.portfolio.total_value * self.risk_params["max_position_size"]:
                return {
                    "status": "rejected",
                    "reason": "Position size exceeds limit",
                    "position_value": position_value,
                    "max_allowed": self.portfolio.total_value * self.risk_params["max_position_size"]
                }
            
            # Execute buy
            self.portfolio.cash -= required_cash
            
            if symbol in self.portfolio.positions:
                # Add to existing position (average in)
                pos = self.portfolio.positions[symbol]
                total_quantity = pos.quantity + quantity
                avg_price = ((pos.quantity * pos.entry_price) + (quantity * price)) / total_quantity
                pos.quantity = total_quantity
                pos.entry_price = avg_price
                pos.current_price = price
                if stop_loss:
                    pos.stop_loss = stop_loss
                if take_profit:
                    pos.take_profit = take_profit
            else:
                # New position
                self.portfolio.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    entry_date=timestamp,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
            trade_record = {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "price": price,
                "value": required_cash,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
            
        else:  # sell
            if symbol not in self.portfolio.positions:
                return {
                    "status": "rejected",
                    "reason": "No position to sell",
                    "symbol": symbol
                }
            
            position = self.portfolio.positions[symbol]
            
            if quantity > position.quantity:
                return {
                    "status": "rejected",
                    "reason": "Insufficient shares",
                    "requested": quantity,
                    "available": position.quantity
                }
            
            # Execute sell
            proceeds = quantity * price
            self.portfolio.cash += proceeds
            
            # Calculate realized P&L
            realized_pnl = (price - position.entry_price) * quantity
            
            if quantity == position.quantity:
                # Close entire position
                del self.portfolio.positions[symbol]
            else:
                # Partial sell
                position.quantity -= quantity
            
            trade_record = {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "action": "sell",
                "quantity": quantity,
                "price": price,
                "value": proceeds,
                "realized_pnl": realized_pnl,
                "pnl_percentage": (realized_pnl / (position.entry_price * quantity)) * 100
            }
        
        # Record trade
        self.trade_history.append(trade_record)
        
        return {
            "status": "executed",
            "trade": trade_record,
            "portfolio_cash": self.portfolio.cash,
            "portfolio_value": self.portfolio.total_value
        }
    
    async def _get_risk_metrics(self) -> Dict:
        """Calculate current portfolio risk metrics"""
        if not self.portfolio.positions:
            return {
                "total_risk": 0,
                "position_risks": [],
                "correlation_risk": 0,
                "concentration_risk": 0,
                "var_95": 0,
                "max_drawdown_risk": 0
            }
        
        # Calculate position-level risks
        position_risks = []
        total_risk_amount = 0
        
        for symbol, position in self.portfolio.positions.items():
            if position.stop_loss:
                risk_per_share = position.current_price - position.stop_loss
                risk_amount = risk_per_share * position.quantity
                risk_percentage = risk_amount / self.portfolio.total_value * 100
            else:
                # Use 2% default risk if no stop loss
                risk_amount = position.market_value * 0.02
                risk_percentage = 2.0
            
            total_risk_amount += risk_amount
            
            position_risks.append({
                "symbol": symbol,
                "risk_amount": risk_amount,
                "risk_percentage": risk_percentage,
                "position_weight": position.market_value / self.portfolio.total_value * 100
            })
        
        # Calculate concentration risk (Herfindahl index)
        weights = [pos.market_value / self.portfolio.total_market_value 
                  for pos in self.portfolio.positions.values()]
        hhi = sum(w**2 for w in weights)
        concentration_risk = hhi
        
        # Simulate correlation risk (in real implementation, would use actual correlations)
        correlation_risk = min(len(self.portfolio.positions) * 0.1, 0.5)
        
        # Calculate VaR (simplified)
        portfolio_volatility = 0.02  # Assumed 2% daily volatility
        var_95 = self.portfolio.total_value * portfolio_volatility * 1.645
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_risk_amount": total_risk_amount,
            "total_risk_percentage": total_risk_amount / self.portfolio.total_value * 100,
            "position_risks": position_risks,
            "concentration_risk": concentration_risk,
            "correlation_risk": correlation_risk,
            "var_95": var_95,
            "var_95_percentage": var_95 / self.portfolio.total_value * 100,
            "risk_limits": {
                "max_position_size": self.risk_params["max_position_size"] * 100,
                "max_risk_per_trade": self.risk_params["max_risk_per_trade"] * 100,
                "max_portfolio_risk": self.risk_params["max_portfolio_risk"] * 100
            }
        }
    
    async def _rebalance_portfolio(self, args: Dict) -> Dict:
        """Generate rebalancing recommendations"""
        target_allocations = args["target_allocations"]
        
        # Normalize target allocations
        total_target = sum(target_allocations.values())
        if total_target > 1.0:
            target_allocations = {k: v/total_target for k, v in target_allocations.items()}
        
        current_allocation = self._calculate_allocation()
        recommendations = []
        
        # Calculate required changes
        for symbol, target_pct in target_allocations.items():
            current_pct = current_allocation.get(symbol, 0) / 100
            diff_pct = target_pct - current_pct
            
            if abs(diff_pct) > 0.01:  # 1% threshold
                # Calculate shares to trade
                target_value = self.portfolio.total_value * target_pct
                current_value = self.portfolio.positions[symbol].market_value if symbol in self.portfolio.positions else 0
                value_diff = target_value - current_value
                
                if symbol in self.portfolio.positions:
                    price = self.portfolio.positions[symbol].current_price
                else:
                    # Simulate price for new positions
                    price = 100 * (1 + random.uniform(-0.5, 0.5))
                
                shares = int(abs(value_diff) / price)
                
                if shares > 0:
                    recommendations.append({
                        "symbol": symbol,
                        "action": "buy" if value_diff > 0 else "sell",
                        "shares": shares,
                        "current_allocation": current_pct * 100,
                        "target_allocation": target_pct * 100,
                        "value_change": value_diff
                    })
        
        # Add sells for positions not in target
        for symbol in self.portfolio.positions:
            if symbol not in target_allocations:
                position = self.portfolio.positions[symbol]
                recommendations.append({
                    "symbol": symbol,
                    "action": "sell",
                    "shares": position.quantity,
                    "current_allocation": current_allocation[symbol],
                    "target_allocation": 0,
                    "value_change": -position.market_value
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_allocation": current_allocation,
            "target_allocation": {k: v*100 for k, v in target_allocations.items()},
            "recommendations": recommendations,
            "estimated_trades": len(recommendations),
            "rebalance_value": sum(abs(r["value_change"]) for r in recommendations)
        }
    
    async def _get_performance_analytics(self, args: Dict) -> Dict:
        """Calculate detailed performance analytics"""
        period_days = args.get("period_days", 30)
        
        if not self.portfolio_history:
            return {
                "error": "Insufficient history for analysis",
                "history_length": 0
            }
        
        # Get relevant history
        cutoff_date = datetime.now() - timedelta(days=period_days)
        relevant_history = [
            h for h in self.portfolio_history 
            if datetime.fromisoformat(h["timestamp"]) > cutoff_date
        ]
        
        if len(relevant_history) < 2:
            return {
                "error": "Insufficient history for analysis",
                "history_length": len(relevant_history)
            }
        
        # Calculate returns
        values = [h["total_value"] for h in relevant_history]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        if not returns:
            return {
                "error": "No returns to analyze",
                "history_length": len(relevant_history)
            }
        
        # Performance metrics
        total_return = (values[-1] - values[0]) / values[0] * 100
        avg_return = np.mean(returns) * 100
        volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = [r - risk_free_rate/252 for r in returns]
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Max drawdown
        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Win rate from trades
        winning_trades = [t for t in self.trade_history if t.get("realized_pnl", 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get("realized_pnl", 0) < 0]
        total_trades = len(winning_trades) + len(losing_trades)
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = np.mean([t["realized_pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["realized_pnl"] for t in losing_trades]) if losing_trades else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "period_days": period_days,
            "data_points": len(relevant_history),
            "performance": {
                "total_return": total_return,
                "average_daily_return": avg_return,
                "annualized_return": avg_return * 252,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown * 100,
                "current_value": self.portfolio.total_value,
                "initial_value": self.portfolio.initial_capital
            },
            "trading_stats": {
                "total_trades": total_trades,
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": win_rate,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0
            }
        }
    
    def _calculate_allocation(self) -> Dict[str, float]:
        """Calculate current portfolio allocation percentages"""
        allocation = {}
        
        if self.portfolio.total_value == 0:
            return allocation
        
        # Cash allocation
        allocation["CASH"] = (self.portfolio.cash / self.portfolio.total_value) * 100
        
        # Position allocations
        for symbol, position in self.portfolio.positions.items():
            allocation[symbol] = (position.market_value / self.portfolio.total_value) * 100
        
        return allocation
    
    async def run(self):
        """Run the MCP server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="goldensignals-portfolio",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

async def main():
    """Main entry point"""
    server = PortfolioManagementServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 