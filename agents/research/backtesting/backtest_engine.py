"""
Backtesting engine for testing trading strategies with historical data.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from agents.orchestration.orchestrator import AgentOrchestrator

class BacktestEngine:
    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        initial_capital: float = 100000.0,
        commission: float = 0.001  # 0.1% commission
    ):
        self.orchestrator = orchestrator
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
        
    def reset(self) -> None:
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        
    def prepare_market_data(
        self,
        prices: pd.Series,
        texts: Optional[List[str]] = None,
        window: int = 100
    ) -> Dict[str, Any]:
        """Prepare market data for agents"""
        return {
            "close_prices": prices.tolist()[-window:],
            "texts": texts or [],
            "timestamp": prices.index[-1].isoformat()
        }
        
    def execute_trade(
        self,
        price: float,
        action: str,
        confidence: float,
        timestamp: datetime
    ) -> None:
        """Execute a trade and update portfolio"""
        if action == "buy" and self.position <= 0:
            # Close short position if exists
            if self.position < 0:
                profit = -self.position * (self.last_price - price)
                self.capital += profit
                self.trades.append({
                    "timestamp": timestamp,
                    "action": "cover",
                    "price": price,
                    "size": -self.position,
                    "profit": profit,
                    "commission": abs(self.position) * price * self.commission
                })
            
            # Open long position
            size = int((self.capital * confidence) / price)
            cost = size * price
            commission = cost * self.commission
            self.capital -= (cost + commission)
            self.position = size
            self.trades.append({
                "timestamp": timestamp,
                "action": "buy",
                "price": price,
                "size": size,
                "cost": cost,
                "commission": commission
            })
            
        elif action == "sell" and self.position >= 0:
            # Close long position if exists
            if self.position > 0:
                profit = self.position * (price - self.last_price)
                self.capital += profit
                self.trades.append({
                    "timestamp": timestamp,
                    "action": "sell",
                    "price": price,
                    "size": self.position,
                    "profit": profit,
                    "commission": self.position * price * self.commission
                })
            
            # Open short position
            size = int((self.capital * confidence) / price)
            cost = size * price
            commission = cost * self.commission
            self.capital -= commission
            self.position = -size
            self.trades.append({
                "timestamp": timestamp,
                "action": "short",
                "price": price,
                "size": size,
                "cost": cost,
                "commission": commission
            })
            
        self.last_price = price
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": self.calculate_equity(price),
            "position": self.position
        })
        
    def calculate_equity(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        position_value = self.position * current_price if self.position != 0 else 0
        return self.capital + position_value
        
    def run(
        self,
        prices: pd.Series,
        texts: Optional[List[str]] = None,
        window: int = 100
    ) -> Dict[str, Any]:
        """Run backtest"""
        self.reset()
        self.last_price = prices[0]
        results = []
        
        for i in range(window, len(prices)):
            market_data = self.prepare_market_data(
                prices[:i+1],
                texts[i:i+1] if texts else None,
                window
            )
            
            decision = self.orchestrator.process_market_data(market_data)
            
            if decision["action"] != "hold" and decision["confidence"] > 0:
                self.execute_trade(
                    price=prices[i],
                    action=decision["action"],
                    confidence=decision["confidence"],
                    timestamp=prices.index[i]
                )
                
            results.append({
                "timestamp": prices.index[i],
                "price": prices[i],
                "action": decision["action"],
                "confidence": decision["confidence"],
                "equity": self.calculate_equity(prices[i])
            })
            
        return self.calculate_statistics(pd.DataFrame(results))
        
    def calculate_statistics(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate backtest statistics"""
        if len(self.trades) == 0:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "trades": []
            }
            
        equity_series = pd.Series([e["equity"] for e in self.equity_curve])
        returns = equity_series.pct_change().dropna()
        
        # Calculate statistics
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return * (252 / len(returns))
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Calculate drawdown
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate trade statistics
        profitable_trades = [t for t in self.trades if t.get("profit", 0) > 0]
        win_rate = len(profitable_trades) / len(self.trades)
        
        gross_profits = sum(t["profit"] for t in profitable_trades)
        gross_losses = sum(t["profit"] for t in self.trades if t.get("profit", 0) <= 0)
        profit_factor = abs(gross_profits / gross_losses) if gross_losses != 0 else float('inf')
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "trades": self.trades,
            "equity_curve": self.equity_curve
        } 