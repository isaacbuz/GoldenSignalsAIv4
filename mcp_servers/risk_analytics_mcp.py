"""
Risk Analytics MCP Server
Provides centralized risk calculations and portfolio monitoring
Issue #193: MCP-4: Build Risk Analytics MCP Server
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from collections import defaultdict
import time
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """Available risk metrics"""
    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    BETA = "beta"
    ALPHA = "alpha"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    STRESS_TEST = "stress_test"


class PortfolioType(Enum):
    """Portfolio types"""
    EQUITY = "equity"
    OPTIONS = "options"
    MIXED = "mixed"
    CRYPTO = "crypto"


class AlertLevel(Enum):
    """Risk alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Position:
    """Represents a portfolio position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    position_type: str  # long, short, call, put
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl(self) -> float:
        if self.position_type == "long":
            return (self.current_price - self.entry_price) * self.quantity
        elif self.position_type == "short":
            return (self.entry_price - self.current_price) * self.quantity
        return 0


@dataclass
class Portfolio:
    """Portfolio structure"""
    id: str
    name: str
    positions: List[Position]
    cash: float
    portfolio_type: PortfolioType
    created_at: datetime
    
    @property
    def total_value(self) -> float:
        return self.cash + sum(pos.market_value for pos in self.positions)
    
    @property
    def total_pnl(self) -> float:
        return sum(pos.pnl for pos in self.positions)


@dataclass
class RiskAlert:
    """Risk alert structure"""
    id: str
    portfolio_id: str
    metric: RiskMetric
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'metric': self.metric.value,
            'level': self.level.value,
            'timestamp': self.timestamp.isoformat()
        }


class RiskAnalyticsMCP:
    """
    MCP Server for centralized risk analytics
    Provides real-time risk metrics, alerts, and portfolio monitoring
    """
    
    def __init__(self):
        self.app = FastAPI(title="Risk Analytics MCP Server")
        
        # Portfolio storage
        self.portfolios: Dict[str, Portfolio] = {}
        
        # Historical data cache
        self.price_history: Dict[str, pd.DataFrame] = {}
        
        # Risk alerts
        self.alerts: List[RiskAlert] = []
        self.alert_thresholds = self._default_thresholds()
        
        # WebSocket clients for real-time alerts
        self.websocket_clients: List[WebSocket] = []
        
        # Metrics
        self.calculation_times: Dict[str, List[float]] = defaultdict(list)
        
        self._setup_routes()
        
        # Start monitoring
        asyncio.create_task(self._monitor_portfolios())
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default risk thresholds"""
        return {
            RiskMetric.VAR.value: {
                AlertLevel.WARNING.value: 0.05,  # 5% VaR
                AlertLevel.CRITICAL.value: 0.10,  # 10% VaR
                AlertLevel.EMERGENCY.value: 0.20  # 20% VaR
            },
            RiskMetric.MAX_DRAWDOWN.value: {
                AlertLevel.WARNING.value: 0.10,
                AlertLevel.CRITICAL.value: 0.20,
                AlertLevel.EMERGENCY.value: 0.30
            },
            RiskMetric.VOLATILITY.value: {
                AlertLevel.WARNING.value: 0.25,  # 25% annualized vol
                AlertLevel.CRITICAL.value: 0.40,
                AlertLevel.EMERGENCY.value: 0.60
            }
        }
    
    def _setup_routes(self):
        """Set up FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "Risk Analytics MCP",
                "status": "active",
                "portfolios": len(self.portfolios),
                "active_alerts": len([a for a in self.alerts if a.level != AlertLevel.INFO])
            }
        
        @self.app.get("/tools")
        async def list_tools():
            """List available risk analytics tools"""
            return {
                "tools": [
                    {
                        "name": "calculate_var",
                        "description": "Calculate Value at Risk for a portfolio",
                        "parameters": {
                            "portfolio_id": "string",
                            "confidence_level": "number (default: 0.95)",
                            "time_horizon": "integer (days, default: 1)",
                            "method": "string (historical/parametric/monte_carlo)"
                        }
                    },
                    {
                        "name": "calculate_portfolio_metrics",
                        "description": "Calculate comprehensive portfolio metrics",
                        "parameters": {
                            "portfolio_id": "string",
                            "benchmark": "string (optional, default: SPY)"
                        }
                    },
                    {
                        "name": "stress_test",
                        "description": "Run stress test scenarios",
                        "parameters": {
                            "portfolio_id": "string",
                            "scenarios": "array[object] (optional, uses defaults)"
                        }
                    },
                    {
                        "name": "position_risk",
                        "description": "Analyze risk for a specific position",
                        "parameters": {
                            "portfolio_id": "string",
                            "symbol": "string"
                        }
                    },
                    {
                        "name": "correlation_matrix",
                        "description": "Calculate portfolio correlation matrix",
                        "parameters": {
                            "portfolio_id": "string",
                            "lookback_days": "integer (default: 252)"
                        }
                    },
                    {
                        "name": "set_alert_threshold",
                        "description": "Set custom alert thresholds",
                        "parameters": {
                            "metric": "string",
                            "level": "string",
                            "threshold": "number"
                        }
                    }
                ]
            }
        
        @self.app.post("/call")
        async def call_tool(request: Dict[str, Any]):
            """Execute a risk analytics tool"""
            tool_name = request.get("tool")
            params = request.get("parameters", {})
            
            try:
                if tool_name == "calculate_var":
                    return await self._calculate_var(params)
                elif tool_name == "calculate_portfolio_metrics":
                    return await self._calculate_portfolio_metrics(params)
                elif tool_name == "stress_test":
                    return await self._run_stress_test(params)
                elif tool_name == "position_risk":
                    return await self._analyze_position_risk(params)
                elif tool_name == "correlation_matrix":
                    return await self._calculate_correlation_matrix(params)
                elif tool_name == "set_alert_threshold":
                    return await self._set_alert_threshold(params)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
                    
            except Exception as e:
                logger.error(f"Error in tool call {tool_name}: {e}")
                return {"error": str(e), "tool": tool_name}
        
        @self.app.post("/portfolio")
        async def create_portfolio(portfolio_data: Dict[str, Any]):
            """Create or update a portfolio"""
            import uuid
            
            portfolio_id = portfolio_data.get('id', str(uuid.uuid4()))
            
            positions = []
            for pos_data in portfolio_data.get('positions', []):
                positions.append(Position(**pos_data))
            
            portfolio = Portfolio(
                id=portfolio_id,
                name=portfolio_data.get('name', f'Portfolio {portfolio_id}'),
                positions=positions,
                cash=portfolio_data.get('cash', 0),
                portfolio_type=PortfolioType(portfolio_data.get('type', 'mixed')),
                created_at=datetime.now()
            )
            
            self.portfolios[portfolio_id] = portfolio
            
            return {
                "portfolio_id": portfolio_id,
                "total_value": portfolio.total_value,
                "position_count": len(positions)
            }
        
        @self.app.get("/alerts")
        async def get_alerts(
            portfolio_id: Optional[str] = None,
            level: Optional[str] = None,
            limit: int = 100
        ):
            """Get risk alerts"""
            alerts = self.alerts
            
            if portfolio_id:
                alerts = [a for a in alerts if a.portfolio_id == portfolio_id]
            
            if level:
                alert_level = AlertLevel(level)
                alerts = [a for a in alerts if a.level == alert_level]
            
            # Sort by timestamp descending
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            return {
                "alerts": [a.to_dict() for a in alerts[:limit]],
                "total": len(alerts)
            }
        
        @self.app.websocket("/ws/alerts")
        async def websocket_alerts(websocket: WebSocket):
            """WebSocket endpoint for real-time alerts"""
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
    
    async def _calculate_var(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Value at Risk"""
        start_time = time.time()
        
        portfolio_id = params.get('portfolio_id')
        confidence_level = params.get('confidence_level', 0.95)
        time_horizon = params.get('time_horizon', 1)
        method = params.get('method', 'historical')
        
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        # Get portfolio returns
        returns = await self._get_portfolio_returns(portfolio, lookback_days=252)
        
        if len(returns) < 30:
            raise ValueError("Insufficient data for VaR calculation")
        
        # Scale returns to time horizon
        scaled_returns = returns * np.sqrt(time_horizon)
        
        # Calculate VaR based on method
        if method == 'historical':
            var = np.percentile(scaled_returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            mean = np.mean(scaled_returns)
            std = np.std(scaled_returns)
            var = mean + std * stats.norm.ppf(1 - confidence_level)
        elif method == 'monte_carlo':
            var = await self._monte_carlo_var(returns, confidence_level, time_horizon)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Calculate CVaR (Expected Shortfall)
        cvar = np.mean(scaled_returns[scaled_returns <= var])
        
        # Record calculation time
        calc_time = time.time() - start_time
        self.calculation_times['var'].append(calc_time)
        
        # Check for alerts
        await self._check_var_alert(portfolio_id, abs(var), confidence_level)
        
        return {
            "portfolio_id": portfolio_id,
            "var": abs(var),  # Return as positive value
            "cvar": abs(cvar),
            "confidence_level": confidence_level,
            "time_horizon_days": time_horizon,
            "method": method,
            "portfolio_value": portfolio.total_value,
            "var_amount": abs(var) * portfolio.total_value,
            "calculation_time_ms": calc_time * 1000
        }
    
    async def _calculate_portfolio_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        start_time = time.time()
        
        portfolio_id = params.get('portfolio_id')
        benchmark = params.get('benchmark', 'SPY')
        
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        # Get returns
        portfolio_returns = await self._get_portfolio_returns(portfolio)
        benchmark_returns = await self._get_benchmark_returns(benchmark, len(portfolio_returns))
        
        # Calculate metrics
        metrics = {
            "total_value": portfolio.total_value,
            "total_pnl": portfolio.total_pnl,
            "return_pct": (portfolio.total_pnl / portfolio.total_value) * 100,
            "volatility": np.std(portfolio_returns) * np.sqrt(252),  # Annualized
            "sharpe_ratio": self._calculate_sharpe_ratio(portfolio_returns),
            "sortino_ratio": self._calculate_sortino_ratio(portfolio_returns),
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
            "beta": self._calculate_beta(portfolio_returns, benchmark_returns),
            "alpha": self._calculate_alpha(portfolio_returns, benchmark_returns),
            "correlation_to_benchmark": np.corrcoef(portfolio_returns, benchmark_returns)[0, 1],
            "position_count": len(portfolio.positions),
            "largest_position": self._get_largest_position(portfolio),
            "concentration_risk": self._calculate_concentration_risk(portfolio)
        }
        
        # Record calculation time
        calc_time = time.time() - start_time
        self.calculation_times['portfolio_metrics'].append(calc_time)
        
        return {
            "portfolio_id": portfolio_id,
            "metrics": metrics,
            "benchmark": benchmark,
            "calculation_time_ms": calc_time * 1000
        }
    
    async def _run_stress_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run stress test scenarios"""
        portfolio_id = params.get('portfolio_id')
        custom_scenarios = params.get('scenarios', [])
        
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        # Default scenarios if none provided
        if not custom_scenarios:
            scenarios = [
                {"name": "Market Crash", "market_shock": -0.20, "vol_shock": 2.0},
                {"name": "Flash Crash", "market_shock": -0.10, "vol_shock": 3.0},
                {"name": "Rate Hike", "market_shock": -0.05, "vol_shock": 1.5},
                {"name": "Black Swan", "market_shock": -0.30, "vol_shock": 4.0},
                {"name": "Sector Rotation", "market_shock": 0.0, "vol_shock": 1.8}
            ]
        else:
            scenarios = custom_scenarios
        
        results = []
        
        for scenario in scenarios:
            # Simulate scenario impact
            scenario_result = await self._simulate_scenario(portfolio, scenario)
            results.append(scenario_result)
        
        # Find worst case
        worst_case = min(results, key=lambda x: x['portfolio_value'])
        
        return {
            "portfolio_id": portfolio_id,
            "current_value": portfolio.total_value,
            "scenarios": results,
            "worst_case": worst_case,
            "survival_probability": self._calculate_survival_probability(results)
        }
    
    async def _analyze_position_risk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk for a specific position"""
        portfolio_id = params.get('portfolio_id')
        symbol = params.get('symbol')
        
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        position = next((p for p in portfolio.positions if p.symbol == symbol), None)
        if not position:
            raise ValueError(f"Position {symbol} not found in portfolio")
        
        # Get position returns
        returns = await self._get_symbol_returns(symbol)
        
        # Calculate position-specific metrics
        position_weight = position.market_value / portfolio.total_value
        position_var = np.percentile(returns, 5) * position.market_value
        
        return {
            "symbol": symbol,
            "position_value": position.market_value,
            "position_pnl": position.pnl,
            "position_weight": position_weight,
            "position_var_95": abs(position_var),
            "volatility": np.std(returns) * np.sqrt(252),
            "contribution_to_portfolio_risk": position_weight * np.std(returns),
            "liquidation_cost_estimate": self._estimate_liquidation_cost(position),
            "recommendations": self._get_position_recommendations(position, position_weight)
        }
    
    async def _calculate_correlation_matrix(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio correlation matrix"""
        portfolio_id = params.get('portfolio_id')
        lookback_days = params.get('lookback_days', 252)
        
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        # Get returns for all positions
        symbols = [p.symbol for p in portfolio.positions]
        returns_data = {}
        
        for symbol in symbols:
            returns_data[symbol] = await self._get_symbol_returns(symbol, lookback_days)
        
        # Create returns DataFrame
        df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    high_correlations.append({
                        "pair": f"{symbols[i]}-{symbols[j]}",
                        "correlation": corr
                    })
        
        return {
            "portfolio_id": portfolio_id,
            "correlation_matrix": corr_matrix.to_dict(),
            "symbols": symbols,
            "high_correlations": high_correlations,
            "diversification_score": self._calculate_diversification_score(corr_matrix)
        }
    
    async def _set_alert_threshold(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set custom alert thresholds"""
        metric = RiskMetric(params.get('metric'))
        level = AlertLevel(params.get('level'))
        threshold = params.get('threshold')
        
        if metric.value not in self.alert_thresholds:
            self.alert_thresholds[metric.value] = {}
        
        self.alert_thresholds[metric.value][level.value] = threshold
        
        return {
            "metric": metric.value,
            "level": level.value,
            "threshold": threshold,
            "status": "updated"
        }
    
    async def _monitor_portfolios(self):
        """Background task to monitor portfolios"""
        while True:
            try:
                for portfolio_id, portfolio in self.portfolios.items():
                    # Calculate key metrics
                    returns = await self._get_portfolio_returns(portfolio, lookback_days=20)
                    
                    if len(returns) > 0:
                        # Check volatility
                        vol = np.std(returns) * np.sqrt(252)
                        await self._check_volatility_alert(portfolio_id, vol)
                        
                        # Check drawdown
                        drawdown = self._calculate_max_drawdown(returns)
                        await self._check_drawdown_alert(portfolio_id, drawdown)
                
                # Sleep for 1 minute
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in portfolio monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _get_portfolio_returns(self, portfolio: Portfolio, lookback_days: int = 252) -> np.ndarray:
        """Get historical returns for portfolio"""
        # Simplified - in production would use actual historical data
        # Generate synthetic returns based on position volatilities
        returns = np.random.normal(0.0005, 0.02, lookback_days)
        return returns
    
    async def _get_symbol_returns(self, symbol: str, lookback_days: int = 252) -> np.ndarray:
        """Get historical returns for a symbol"""
        # Simplified - in production would use actual historical data
        returns = np.random.normal(0.0005, 0.025, lookback_days)
        return returns
    
    async def _get_benchmark_returns(self, benchmark: str, days: int) -> np.ndarray:
        """Get benchmark returns"""
        # Simplified - in production would use actual benchmark data
        returns = np.random.normal(0.0004, 0.015, days)
        return returns
    
    async def _monte_carlo_var(self, returns: np.ndarray, confidence_level: float, 
                             time_horizon: int, simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Run simulations
        simulated_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * np.sqrt(time_horizon),
            simulations
        )
        
        # Calculate VaR
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        return var
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.001
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_beta(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate portfolio beta"""
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        return covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    def _calculate_alpha(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray,
                       risk_free_rate: float = 0.02) -> float:
        """Calculate portfolio alpha"""
        portfolio_return = np.mean(portfolio_returns) * 252
        benchmark_return = np.mean(benchmark_returns) * 252
        beta = self._calculate_beta(portfolio_returns, benchmark_returns)
        
        alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
        return alpha
    
    def _get_largest_position(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Get largest position in portfolio"""
        if not portfolio.positions:
            return {"symbol": "None", "weight": 0}
        
        largest = max(portfolio.positions, key=lambda p: p.market_value)
        return {
            "symbol": largest.symbol,
            "weight": largest.market_value / portfolio.total_value
        }
    
    def _calculate_concentration_risk(self, portfolio: Portfolio) -> float:
        """Calculate concentration risk (Herfindahl index)"""
        if not portfolio.positions:
            return 0
        
        weights = [p.market_value / portfolio.total_value for p in portfolio.positions]
        return sum(w**2 for w in weights)
    
    async def _simulate_scenario(self, portfolio: Portfolio, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a stress scenario"""
        market_shock = scenario.get('market_shock', 0)
        vol_shock = scenario.get('vol_shock', 1)
        
        # Apply shocks to portfolio
        shocked_value = portfolio.total_value * (1 + market_shock)
        
        # Estimate increased volatility impact
        normal_vol = 0.15  # 15% annual volatility
        shocked_vol = normal_vol * vol_shock
        
        # Additional loss from volatility
        vol_impact = shocked_value * shocked_vol * 0.1  # Simplified
        
        final_value = shocked_value - vol_impact
        loss = portfolio.total_value - final_value
        
        return {
            "scenario": scenario['name'],
            "portfolio_value": final_value,
            "loss": loss,
            "loss_pct": (loss / portfolio.total_value) * 100,
            "survival": final_value > portfolio.total_value * 0.5
        }
    
    def _calculate_survival_probability(self, stress_results: List[Dict[str, Any]]) -> float:
        """Calculate probability of surviving stress scenarios"""
        survivals = [r['survival'] for r in stress_results]
        return sum(survivals) / len(survivals) if survivals else 0
    
    def _estimate_liquidation_cost(self, position: Position) -> float:
        """Estimate cost to liquidate position"""
        # Simplified - based on position size
        # Larger positions have higher liquidation costs
        size_factor = min(position.market_value / 1000000, 1)  # Cap at $1M
        base_cost = 0.001  # 10 bps base cost
        
        return position.market_value * (base_cost + size_factor * 0.004)
    
    def _get_position_recommendations(self, position: Position, weight: float) -> List[str]:
        """Get recommendations for position management"""
        recommendations = []
        
        if weight > 0.25:
            recommendations.append(f"Position too concentrated ({weight:.1%}). Consider reducing.")
        
        if position.pnl < -position.market_value * 0.20:
            recommendations.append("Position down >20%. Review stop-loss strategy.")
        
        if position.position_type == "short" and position.pnl < 0:
            recommendations.append("Short position moving against you. Monitor closely.")
        
        return recommendations
    
    def _calculate_diversification_score(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate portfolio diversification score (0-1)"""
        # Average correlation excluding diagonal
        n = len(corr_matrix)
        if n <= 1:
            return 0
        
        total_corr = 0
        count = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    total_corr += abs(corr_matrix.iloc[i, j])
                    count += 1
        
        avg_corr = total_corr / count if count > 0 else 1
        
        # Convert to score (lower correlation = higher score)
        return max(0, 1 - avg_corr)
    
    async def _check_var_alert(self, portfolio_id: str, var_value: float, confidence: float):
        """Check if VaR breaches alert thresholds"""
        thresholds = self.alert_thresholds.get(RiskMetric.VAR.value, {})
        
        for level, threshold in thresholds.items():
            if var_value >= threshold:
                alert = RiskAlert(
                    id=str(len(self.alerts) + 1),
                    portfolio_id=portfolio_id,
                    metric=RiskMetric.VAR,
                    level=AlertLevel(level),
                    message=f"VaR ({confidence:.0%}) exceeds threshold: {var_value:.2%} > {threshold:.2%}",
                    value=var_value,
                    threshold=threshold,
                    timestamp=datetime.now()
                )
                
                await self._send_alert(alert)
                break
    
    async def _check_volatility_alert(self, portfolio_id: str, volatility: float):
        """Check if volatility breaches alert thresholds"""
        thresholds = self.alert_thresholds.get(RiskMetric.VOLATILITY.value, {})
        
        for level, threshold in thresholds.items():
            if volatility >= threshold:
                alert = RiskAlert(
                    id=str(len(self.alerts) + 1),
                    portfolio_id=portfolio_id,
                    metric=RiskMetric.VOLATILITY,
                    level=AlertLevel(level),
                    message=f"Volatility exceeds threshold: {volatility:.2%} > {threshold:.2%}",
                    value=volatility,
                    threshold=threshold,
                    timestamp=datetime.now()
                )
                
                await self._send_alert(alert)
                break
    
    async def _check_drawdown_alert(self, portfolio_id: str, drawdown: float):
        """Check if drawdown breaches alert thresholds"""
        thresholds = self.alert_thresholds.get(RiskMetric.MAX_DRAWDOWN.value, {})
        
        for level, threshold in thresholds.items():
            if drawdown >= threshold:
                alert = RiskAlert(
                    id=str(len(self.alerts) + 1),
                    portfolio_id=portfolio_id,
                    metric=RiskMetric.MAX_DRAWDOWN,
                    level=AlertLevel(level),
                    message=f"Drawdown exceeds threshold: {drawdown:.2%} > {threshold:.2%}",
                    value=drawdown,
                    threshold=threshold,
                    timestamp=datetime.now()
                )
                
                await self._send_alert(alert)
                break
    
    async def _send_alert(self, alert: RiskAlert):
        """Send alert to all connected clients"""
        self.alerts.append(alert)
        
        # Send to WebSocket clients
        alert_data = alert.to_dict()
        for client in self.websocket_clients:
            try:
                await client.send_json({
                    "type": "risk_alert",
                    "alert": alert_data
                })
            except Exception as e:
                logger.error(f"Failed to send alert via WebSocket: {e}")
        
        # Log critical alerts
        if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            logger.warning(f"RISK ALERT: {alert.message}")


# Demo function
async def demo_risk_analytics_mcp():
    """Demonstrate Risk Analytics MCP functionality"""
    import uvicorn
    
    logger.info("Starting Risk Analytics MCP Server demo...")
    
    # Create server
    server = RiskAnalyticsMCP()
    
    # Add sample portfolio
    sample_portfolio = {
        "name": "Demo Portfolio",
        "cash": 100000,
        "type": "mixed",
        "positions": [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "entry_price": 150,
                "current_price": 155,
                "position_type": "long"
            },
            {
                "symbol": "GOOGL",
                "quantity": 50,
                "entry_price": 2800,
                "current_price": 2750,
                "position_type": "long"
            },
            {
                "symbol": "TSLA",
                "quantity": -20,  # Short position
                "entry_price": 900,
                "current_price": 920,
                "position_type": "short"
            }
        ]
    }
    
    # Create portfolio
    response = await server.app.app.state.create_portfolio(sample_portfolio)
    logger.info(f"Created demo portfolio: {response}")
    
    # Run server
    config = uvicorn.Config(
        app=server.app,
        host="0.0.0.0",
        port=8193,
        log_level="info"
    )
    
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


if __name__ == "__main__":
    asyncio.run(demo_risk_analytics_mcp()) 