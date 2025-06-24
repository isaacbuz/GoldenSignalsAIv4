"""
Risk Management Simulator - Comprehensive risk testing framework
Tests portfolio risk, VaR calculations, stress scenarios, and circuit breakers
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        if self.entry_price == 0:
            return 0
        return (self.current_price - self.entry_price) / self.entry_price


@dataclass
class Portfolio:
    """Portfolio state tracking"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    historical_returns: List[float] = field(default_factory=list)
    historical_values: List[float] = field(default_factory=list)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    
    @property
    def total_value(self) -> float:
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value
    
    @property
    def exposure(self) -> float:
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    @property
    def leverage(self) -> float:
        if self.total_value == 0:
            return 0
        return self.exposure / self.total_value


class RiskMetric(Enum):
    """Types of risk metrics"""
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    BETA = "beta"
    CORRELATION = "correlation"


@dataclass
class StressScenario:
    """Defines a stress test scenario"""
    name: str
    description: str
    market_shock: Dict[str, float]  # Symbol -> price change %
    volatility_multiplier: float
    correlation_breakdown: bool
    duration_days: int
    probability: float = 0.01  # Tail event probability


@dataclass
class CircuitBreaker:
    """Circuit breaker configuration"""
    name: str
    metric: str  # 'loss', 'volatility', 'drawdown'
    threshold: float
    action: str  # 'halt', 'reduce_position', 'close_all'
    cooldown_minutes: int
    is_active: bool = True
    last_triggered: Optional[datetime] = None


class RiskManagementSimulator:
    """
    Comprehensive risk management testing framework
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Risk parameters
        self.var_confidence_levels = self.config.get('var_confidence_levels', [0.95, 0.99])
        self.lookback_days = self.config.get('lookback_days', 252)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        
        # Circuit breakers
        self.circuit_breakers: List[CircuitBreaker] = []
        self._init_default_circuit_breakers()
        
        # Risk history
        self.risk_history: List[Dict[str, Any]] = []
        self.breach_history: List[Dict[str, Any]] = []
        
    def _init_default_circuit_breakers(self):
        """Initialize default circuit breakers"""
        self.circuit_breakers = [
            CircuitBreaker(
                name="Daily Loss Limit",
                metric="loss",
                threshold=0.05,  # 5% daily loss
                action="halt",
                cooldown_minutes=60
            ),
            CircuitBreaker(
                name="Volatility Spike",
                metric="volatility",
                threshold=3.0,  # 3x normal volatility
                action="reduce_position",
                cooldown_minutes=30
            ),
            CircuitBreaker(
                name="Drawdown Limit",
                metric="drawdown",
                threshold=0.10,  # 10% drawdown
                action="close_all",
                cooldown_minutes=120
            )
        ]
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            VaR value (negative number representing potential loss)
        """
        if len(returns) == 0:
            return 0
        
        if method == 'historical':
            # Historical simulation
            var = np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Assume normal distribution
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean = np.mean(returns)
            std = np.std(returns)
            simulated_returns = np.random.normal(mean, std, 10000)
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return var
    
    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        
        Returns:
            Average loss beyond VaR
        """
        if len(returns) == 0:
            return 0
        
        var = self.calculate_var(returns, confidence_level)
        # Get returns worse than VaR
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return np.mean(tail_returns)
    
    def calculate_risk_metrics(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for a portfolio"""
        if len(portfolio.historical_returns) < 2:
            return {}
        
        returns = np.array(portfolio.historical_returns)
        
        metrics = {}
        
        # VaR and CVaR
        for confidence in self.var_confidence_levels:
            var_key = f"var_{int(confidence*100)}"
            cvar_key = f"cvar_{int(confidence*100)}"
            
            metrics[var_key] = self.calculate_var(returns, confidence)
            metrics[cvar_key] = self.calculate_cvar(returns, confidence)
        
        # Drawdown
        cumulative_returns = np.cumprod(1 + returns) - 1
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        metrics['max_drawdown'] = drawdown.min()
        metrics['current_drawdown'] = drawdown[-1] if len(drawdown) > 0 else 0
        
        # Risk-adjusted returns
        if np.std(returns) > 0:
            metrics['sharpe_ratio'] = (np.mean(returns) - self.risk_free_rate/252) / np.std(returns) * np.sqrt(252)
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    metrics['sortino_ratio'] = (np.mean(returns) - self.risk_free_rate/252) / downside_std * np.sqrt(252)
                else:
                    metrics['sortino_ratio'] = np.inf if np.mean(returns) > self.risk_free_rate/252 else 0
            else:
                metrics['sortino_ratio'] = np.inf
        else:
            metrics['sharpe_ratio'] = 0
            metrics['sortino_ratio'] = 0
        
        # Portfolio metrics
        metrics['total_value'] = portfolio.total_value
        metrics['exposure'] = portfolio.exposure
        metrics['leverage'] = portfolio.leverage
        metrics['position_count'] = len(portfolio.positions)
        
        # Volatility
        metrics['volatility_daily'] = np.std(returns)
        metrics['volatility_annual'] = np.std(returns) * np.sqrt(252)
        
        return metrics
    
    def run_stress_test(
        self,
        portfolio: Portfolio,
        scenario: StressScenario,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Run a stress test scenario on the portfolio
        
        Args:
            portfolio: Current portfolio
            scenario: Stress scenario to test
            market_data: Historical market data for correlation
            
        Returns:
            Stress test results
        """
        results = {
            'scenario': scenario.name,
            'description': scenario.description,
            'initial_value': portfolio.total_value,
            'positions': {}
        }
        
        # Simulate shocked portfolio value
        shocked_value = portfolio.cash
        total_loss = 0
        
        for symbol, position in portfolio.positions.items():
            # Apply market shock
            if symbol in scenario.market_shock:
                shock = scenario.market_shock[symbol]
            else:
                # Use average shock for unlisted symbols
                shock = np.mean(list(scenario.market_shock.values()))
            
            # Add volatility component
            if symbol in market_data:
                historical_vol = market_data[symbol]['close'].pct_change().std()
                vol_shock = historical_vol * scenario.volatility_multiplier * np.random.normal()
                total_shock = shock + vol_shock
            else:
                total_shock = shock
            
            # Calculate shocked price
            shocked_price = position.current_price * (1 + total_shock)
            position_loss = position.quantity * (position.current_price - shocked_price)
            
            shocked_value += position.quantity * shocked_price
            total_loss += position_loss
            
            results['positions'][symbol] = {
                'shock_percentage': total_shock * 100,
                'shocked_price': shocked_price,
                'position_loss': position_loss,
                'loss_percentage': position_loss / (position.quantity * position.current_price) * 100
            }
        
        results['final_value'] = shocked_value
        results['total_loss'] = total_loss
        results['loss_percentage'] = (total_loss / portfolio.total_value) * 100
        results['var_breach'] = total_loss < portfolio.total_value * self.calculate_var(portfolio.historical_returns, 0.99)
        
        # Calculate correlations under stress
        if scenario.correlation_breakdown and len(portfolio.positions) > 1:
            correlations = self._calculate_stress_correlations(
                portfolio, market_data, scenario.volatility_multiplier
            )
            results['stress_correlations'] = correlations
        
        return results
    
    def _calculate_stress_correlations(
        self,
        portfolio: Portfolio,
        market_data: Dict[str, pd.DataFrame],
        volatility_multiplier: float
    ) -> Dict[str, float]:
        """Calculate correlations under stressed conditions"""
        symbols = list(portfolio.positions.keys())
        
        if len(symbols) < 2:
            return {}
        
        # Get returns for all symbols
        returns_data = {}
        for symbol in symbols:
            if symbol in market_data:
                returns = market_data[symbol]['close'].pct_change().dropna()
                # Amplify tail movements
                returns_data[symbol] = returns * (1 + (volatility_multiplier - 1) * (abs(returns) > returns.std() * 2))
        
        if len(returns_data) < 2:
            return {}
        
        # Calculate pairwise correlations
        returns_df = pd.DataFrame(returns_data)
        correlations = returns_df.corr()
        
        # Extract unique pairs
        correlation_pairs = {}
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pair = f"{symbols[i]}-{symbols[j]}"
                correlation_pairs[pair] = correlations.iloc[i, j]
        
        return correlation_pairs
    
    def check_circuit_breakers(
        self,
        portfolio: Portfolio,
        market_data: Dict[str, pd.DataFrame],
        current_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Check if any circuit breakers should be triggered
        
        Returns:
            List of triggered circuit breakers with recommended actions
        """
        triggered = []
        risk_metrics = self.calculate_risk_metrics(portfolio)
        
        for breaker in self.circuit_breakers:
            if not breaker.is_active:
                continue
            
            # Check cooldown
            if breaker.last_triggered:
                cooldown_end = breaker.last_triggered + timedelta(minutes=breaker.cooldown_minutes)
                if current_time < cooldown_end:
                    continue
            
            triggered_flag = False
            current_value = None
            
            # Check different metrics
            if breaker.metric == 'loss':
                if len(portfolio.historical_returns) > 0:
                    daily_return = portfolio.historical_returns[-1]
                    if daily_return < -breaker.threshold:
                        triggered_flag = True
                        current_value = daily_return
            
            elif breaker.metric == 'volatility':
                if 'volatility_daily' in risk_metrics:
                    # Compare to rolling average volatility
                    if len(portfolio.historical_returns) > 20:
                        recent_returns = portfolio.historical_returns[-20:]
                        avg_vol = np.std(recent_returns[:-1])
                        current_vol = risk_metrics['volatility_daily']
                        if avg_vol > 0 and current_vol / avg_vol > breaker.threshold:
                            triggered_flag = True
                            current_value = current_vol / avg_vol
            
            elif breaker.metric == 'drawdown':
                if 'current_drawdown' in risk_metrics:
                    if risk_metrics['current_drawdown'] < -breaker.threshold:
                        triggered_flag = True
                        current_value = risk_metrics['current_drawdown']
            
            if triggered_flag:
                breaker.last_triggered = current_time
                
                triggered.append({
                    'breaker': breaker.name,
                    'metric': breaker.metric,
                    'threshold': breaker.threshold,
                    'current_value': current_value,
                    'action': breaker.action,
                    'timestamp': current_time,
                    'portfolio_value': portfolio.total_value,
                    'recommended_actions': self._get_recommended_actions(breaker, portfolio)
                })
                
                # Log breach
                self.breach_history.append({
                    'timestamp': current_time,
                    'breaker': breaker.name,
                    'value': current_value,
                    'action': breaker.action
                })
        
        return triggered
    
    def _get_recommended_actions(self, breaker: CircuitBreaker, portfolio: Portfolio) -> List[str]:
        """Get recommended actions for triggered circuit breaker"""
        actions = []
        
        if breaker.action == 'halt':
            actions.append("Halt all trading immediately")
            actions.append("Review current positions and market conditions")
            actions.append(f"Trading can resume after {breaker.cooldown_minutes} minutes")
        
        elif breaker.action == 'reduce_position':
            actions.append("Reduce position sizes by 50%")
            actions.append("Focus on highest risk positions first")
            large_positions = sorted(
                portfolio.positions.items(),
                key=lambda x: abs(x[1].market_value),
                reverse=True
            )[:3]
            for symbol, pos in large_positions:
                actions.append(f"Reduce {symbol}: {pos.quantity} shares (${pos.market_value:,.2f})")
        
        elif breaker.action == 'close_all':
            actions.append("Close all positions immediately")
            actions.append("Move to 100% cash")
            total_value = sum(pos.market_value for pos in portfolio.positions.values())
            actions.append(f"Total exposure to close: ${total_value:,.2f}")
        
        return actions
    
    def generate_risk_report(
        self,
        portfolio: Portfolio,
        market_data: Dict[str, pd.DataFrame],
        stress_scenarios: List[StressScenario]
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        report = {
            'timestamp': datetime.now(),
            'portfolio_summary': {
                'total_value': portfolio.total_value,
                'cash': portfolio.cash,
                'exposure': portfolio.exposure,
                'leverage': portfolio.leverage,
                'position_count': len(portfolio.positions)
            }
        }
        
        # Current risk metrics
        report['risk_metrics'] = self.calculate_risk_metrics(portfolio)
        
        # Position-level risk
        position_risks = {}
        for symbol, position in portfolio.positions.items():
            position_risks[symbol] = {
                'market_value': position.market_value,
                'pnl_percentage': position.pnl_percentage * 100,
                'weight': position.market_value / portfolio.total_value * 100,
                'var_contribution': self._calculate_position_var_contribution(
                    position, portfolio, market_data.get(symbol)
                )
            }
        report['position_risks'] = position_risks
        
        # Stress test results
        stress_results = []
        for scenario in stress_scenarios:
            result = self.run_stress_test(portfolio, scenario, market_data)
            stress_results.append(result)
        report['stress_tests'] = stress_results
        
        # Risk limit compliance
        limit_compliance = {}
        for limit_name, limit_value in portfolio.risk_limits.items():
            current_value = report['risk_metrics'].get(limit_name, 0)
            limit_compliance[limit_name] = {
                'limit': limit_value,
                'current': current_value,
                'utilization': abs(current_value / limit_value) * 100 if limit_value != 0 else 0,
                'breach': abs(current_value) > abs(limit_value)
            }
        report['limit_compliance'] = limit_compliance
        
        # Circuit breaker status
        breaker_status = []
        for breaker in self.circuit_breakers:
            status = {
                'name': breaker.name,
                'active': breaker.is_active,
                'last_triggered': breaker.last_triggered.isoformat() if breaker.last_triggered else None,
                'ready': True
            }
            
            if breaker.last_triggered:
                cooldown_end = breaker.last_triggered + timedelta(minutes=breaker.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    status['ready'] = False
                    status['cooldown_remaining'] = (cooldown_end - datetime.now()).total_seconds() / 60
            
            breaker_status.append(status)
        report['circuit_breakers'] = breaker_status
        
        # Risk recommendations
        report['recommendations'] = self._generate_risk_recommendations(report)
        
        return report
    
    def _calculate_position_var_contribution(
        self,
        position: Position,
        portfolio: Portfolio,
        position_data: Optional[pd.DataFrame]
    ) -> float:
        """Calculate position's contribution to portfolio VaR"""
        if position_data is None or len(portfolio.historical_returns) < 10:
            # Fallback: use position weight
            return position.market_value / portfolio.total_value * 100
        
        # Calculate marginal VaR
        position_returns = position_data['close'].pct_change().dropna()
        if len(position_returns) < 10:
            return position.market_value / portfolio.total_value * 100
        
        # Align returns
        recent_position_returns = position_returns.tail(len(portfolio.historical_returns))
        portfolio_returns = np.array(portfolio.historical_returns)
        
        if len(recent_position_returns) != len(portfolio_returns):
            return position.market_value / portfolio.total_value * 100
        
        # Calculate correlation and contribution
        correlation = np.corrcoef(recent_position_returns, portfolio_returns)[0, 1]
        position_weight = position.market_value / portfolio.total_value
        position_vol = np.std(recent_position_returns)
        portfolio_vol = np.std(portfolio_returns)
        
        if portfolio_vol > 0:
            var_contribution = position_weight * correlation * position_vol / portfolio_vol * 100
        else:
            var_contribution = position_weight * 100
        
        return var_contribution
    
    def _generate_risk_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable risk recommendations based on report"""
        recommendations = []
        
        metrics = report['risk_metrics']
        
        # Check VaR levels
        if 'var_99' in metrics and metrics['var_99'] < -0.10:
            recommendations.append("High VaR detected: Consider reducing position sizes or hedging")
        
        # Check leverage
        if metrics.get('leverage', 0) > 2:
            recommendations.append(f"Leverage at {metrics['leverage']:.1f}x: Consider deleveraging")
        
        # Check drawdown
        if metrics.get('current_drawdown', 0) < -0.05:
            recommendations.append(f"In drawdown ({metrics['current_drawdown']:.1%}): Review stop-loss levels")
        
        # Check concentration
        position_risks = report['position_risks']
        for symbol, risk in position_risks.items():
            if risk['weight'] > 20:
                recommendations.append(f"{symbol} is {risk['weight']:.1f}% of portfolio: Consider diversifying")
        
        # Check Sharpe ratio
        if metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("Low risk-adjusted returns: Review strategy effectiveness")
        
        # Check stress test results
        for stress_test in report['stress_tests']:
            if stress_test['loss_percentage'] > 15:
                recommendations.append(
                    f"{stress_test['scenario']} could cause {stress_test['loss_percentage']:.1f}% loss: "
                    f"Consider tail risk hedging"
                )
        
        # Check limit breaches
        for limit_name, compliance in report['limit_compliance'].items():
            if compliance['breach']:
                recommendations.append(
                    f"{limit_name} limit breached: {compliance['current']:.2f} vs {compliance['limit']:.2f} limit"
                )
        
        return recommendations if recommendations else ["Portfolio risk levels are within acceptable ranges"]


# Example usage
def demo_risk_management():
    """Demonstrate risk management capabilities"""
    # Create risk simulator
    risk_sim = RiskManagementSimulator({
        'var_confidence_levels': [0.95, 0.99],
        'lookback_days': 252,
        'risk_free_rate': 0.02
    })
    
    # Create sample portfolio
    portfolio = Portfolio(
        cash=50000,
        positions={
            'AAPL': Position('AAPL', 100, 150, 155, datetime.now()),
            'GOOGL': Position('GOOGL', 50, 2800, 2750, datetime.now()),
            'MSFT': Position('MSFT', 75, 300, 310, datetime.now())
        },
        historical_returns=[0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005, -0.015, 0.01, -0.003],
        risk_limits={
            'var_95': -0.05,
            'max_drawdown': -0.10,
            'leverage': 2.0
        }
    )
    
    # Calculate risk metrics
    print("Risk Management Demo")
    print("=" * 50)
    
    metrics = risk_sim.calculate_risk_metrics(portfolio)
    print("\nRisk Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Define stress scenarios
    stress_scenarios = [
        StressScenario(
            name="Market Crash",
            description="2008-style financial crisis",
            market_shock={'AAPL': -0.30, 'GOOGL': -0.35, 'MSFT': -0.25},
            volatility_multiplier=3.0,
            correlation_breakdown=True,
            duration_days=30
        ),
        StressScenario(
            name="Tech Selloff",
            description="Dot-com bubble burst scenario",
            market_shock={'AAPL': -0.40, 'GOOGL': -0.45, 'MSFT': -0.35},
            volatility_multiplier=2.5,
            correlation_breakdown=True,
            duration_days=90
        ),
        StressScenario(
            name="Flash Crash",
            description="Sudden algorithmic selling",
            market_shock={'AAPL': -0.10, 'GOOGL': -0.12, 'MSFT': -0.08},
            volatility_multiplier=5.0,
            correlation_breakdown=False,
            duration_days=1
        )
    ]
    
    # Generate mock market data for stress testing
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    market_data = {}
    
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        prices = 100 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.02))
        market_data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    # Run stress tests
    print("\nStress Test Results:")
    for scenario in stress_scenarios:
        result = risk_sim.run_stress_test(portfolio, scenario, market_data)
        print(f"\n{scenario.name}:")
        print(f"  Initial Value: ${result['initial_value']:,.2f}")
        print(f"  Final Value: ${result['final_value']:,.2f}")
        print(f"  Total Loss: ${abs(result['total_loss']):,.2f} ({result['loss_percentage']:.1f}%)")
        print(f"  VaR Breach: {result['var_breach']}")
    
    # Check circuit breakers
    print("\nCircuit Breaker Check:")
    triggered = risk_sim.check_circuit_breakers(portfolio, market_data, datetime.now())
    
    if triggered:
        for breach in triggered:
            print(f"\n⚠️  {breach['breaker']} TRIGGERED!")
            print(f"  Current Value: {breach['current_value']:.4f}")
            print(f"  Action: {breach['action']}")
            print("  Recommendations:")
            for rec in breach['recommended_actions']:
                print(f"    - {rec}")
    else:
        print("  ✅ No circuit breakers triggered")
    
    # Generate risk report
    report = risk_sim.generate_risk_report(portfolio, market_data, stress_scenarios)
    
    print("\nRisk Report Summary:")
    print(f"  Portfolio Value: ${report['portfolio_summary']['total_value']:,.2f}")
    print(f"  Leverage: {report['portfolio_summary']['leverage']:.2f}x")
    print("\n  Recommendations:")
    for rec in report['recommendations']:
        print(f"    - {rec}")
    
    return risk_sim, portfolio, report


if __name__ == "__main__":
    demo_risk_management() 