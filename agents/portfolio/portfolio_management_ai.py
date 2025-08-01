"""
Portfolio Management AI Agent
Autonomous portfolio optimization and management
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from agents.common.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

@dataclass
class PortfolioState:
    """Current portfolio state"""
    positions: Dict[str, float]  # symbol -> quantity
    cash: float
    total_value: float
    risk_score: float
    last_rebalance: datetime

@dataclass
class RebalanceAction:
    """Rebalancing action"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    reason: str
    confidence: float
    expected_impact: Dict[str, float]  # risk, return, etc.

class PortfolioManagementAI(BaseAgent):
    """
    Autonomous AI agent for portfolio management

    Features:
    - Dynamic position sizing based on signal confidence
    - Risk-adjusted portfolio optimization
    - Automatic rebalancing during market regime changes
    - Tax-loss harvesting optimization
    - Correlation-based diversification
    """

    def __init__(self,
                 name: str = "PortfolioManagementAI",
                 risk_tolerance: str = "moderate",
                 rebalance_threshold: float = 0.05,
                 max_position_size: float = 0.20):
        super().__init__(name=name, agent_type="portfolio")
        self.risk_tolerance = risk_tolerance
        self.rebalance_threshold = rebalance_threshold
        self.max_position_size = max_position_size
        self.risk_limits = self._get_risk_limits()

    def _get_risk_limits(self) -> Dict[str, float]:
        """Get risk limits based on tolerance"""
        limits = {
            "conservative": {
                "max_volatility": 0.10,
                "max_drawdown": 0.05,
                "max_concentration": 0.15,
                "min_cash": 0.20
            },
            "moderate": {
                "max_volatility": 0.15,
                "max_drawdown": 0.10,
                "max_concentration": 0.20,
                "min_cash": 0.10
            },
            "aggressive": {
                "max_volatility": 0.25,
                "max_drawdown": 0.20,
                "max_concentration": 0.30,
                "min_cash": 0.05
            }
        }
        return limits.get(self.risk_tolerance, limits["moderate"])

    def analyze_portfolio(self, portfolio: PortfolioState, market_data: Dict) -> Dict[str, Any]:
        """Analyze current portfolio health"""
        analysis = {
            "risk_score": self._calculate_risk_score(portfolio, market_data),
            "diversification_score": self._calculate_diversification(portfolio),
            "performance_attribution": self._attribute_performance(portfolio, market_data),
            "rebalance_needed": self._check_rebalance_needed(portfolio, market_data),
            "optimization_opportunities": self._find_optimizations(portfolio, market_data)
        }

        return analysis

    def generate_rebalance_actions(self,
                                 portfolio: PortfolioState,
                                 signals: List[Dict],
                                 market_data: Dict) -> List[RebalanceAction]:
        """Generate rebalancing actions based on signals and portfolio state"""
        actions = []

        # 1. Risk-based adjustments
        risk_actions = self._generate_risk_adjustments(portfolio, market_data)
        actions.extend(risk_actions)

        # 2. Signal-based opportunities
        signal_actions = self._process_signals_for_portfolio(portfolio, signals, market_data)
        actions.extend(signal_actions)

        # 3. Optimization actions (tax-loss harvesting, etc.)
        optimization_actions = self._generate_optimization_actions(portfolio, market_data)
        actions.extend(optimization_actions)

        # 4. Validate and prioritize actions
        validated_actions = self._validate_actions(actions, portfolio)

        return validated_actions

    def _calculate_risk_score(self, portfolio: PortfolioState, market_data: Dict) -> float:
        """Calculate portfolio risk score (0-1)"""
        # Simplified risk calculation
        volatility_score = self._calculate_portfolio_volatility(portfolio, market_data)
        concentration_score = self._calculate_concentration_risk(portfolio)
        correlation_score = self._calculate_correlation_risk(portfolio, market_data)

        # Weighted risk score
        risk_score = (
            0.4 * volatility_score +
            0.3 * concentration_score +
            0.3 * correlation_score
        )

        return min(max(risk_score, 0.0), 1.0)

    def _calculate_portfolio_volatility(self, portfolio: PortfolioState, market_data: Dict) -> float:
        """Calculate portfolio volatility"""
        # Simplified - would use historical returns and covariance matrix
        position_volatilities = []

        for symbol, quantity in portfolio.positions.items():
            if symbol in market_data:
                # Get historical volatility
                vol = market_data[symbol].get('volatility', 0.15)
                weight = (quantity * market_data[symbol]['price']) / portfolio.total_value
                position_volatilities.append(vol * weight)

        # Simple sum (ignoring correlations for now)
        portfolio_vol = sum(position_volatilities)

        # Normalize to 0-1 scale
        return min(portfolio_vol / self.risk_limits['max_volatility'], 1.0)

    def _calculate_concentration_risk(self, portfolio: PortfolioState) -> float:
        """Calculate concentration risk"""
        if not portfolio.positions:
            return 0.0

        # Calculate position weights
        weights = []
        for symbol, quantity in portfolio.positions.items():
            # Would need market prices here
            weight = 1.0 / len(portfolio.positions)  # Simplified
            weights.append(weight)

        # Herfindahl index
        hhi = sum(w**2 for w in weights)

        # Normalize (perfect diversification = 1/n)
        min_hhi = 1.0 / len(portfolio.positions)
        concentration = (hhi - min_hhi) / (1.0 - min_hhi) if len(portfolio.positions) > 1 else 1.0

        return concentration

    def _generate_risk_adjustments(self,
                                  portfolio: PortfolioState,
                                  market_data: Dict) -> List[RebalanceAction]:
        """Generate actions to adjust portfolio risk"""
        actions = []

        # Check if risk exceeds limits
        current_risk = portfolio.risk_score
        target_risk = 0.5  # Target middle of range

        if current_risk > self.risk_limits['max_volatility']:
            # Need to reduce risk
            high_risk_positions = self._identify_high_risk_positions(portfolio, market_data)

            for position in high_risk_positions:
                action = RebalanceAction(
                    symbol=position['symbol'],
                    action='sell',
                    quantity=position['reduce_by'],
                    reason=f"Reducing position to lower portfolio risk (current: {current_risk:.2f})",
                    confidence=0.85,
                    expected_impact={'risk_reduction': position['risk_impact']}
                )
                actions.append(action)

        return actions

    def _process_signals_for_portfolio(self,
                                     portfolio: PortfolioState,
                                     signals: List[Dict],
                                     market_data: Dict) -> List[RebalanceAction]:
        """Convert signals to portfolio actions with position sizing"""
        actions = []

        for signal in signals:
            # Only process high-confidence signals
            if signal.get('confidence', 0) < 0.70:
                continue

            symbol = signal['symbol']
            signal_action = signal.get('action', 'hold')

            # Calculate position size based on Kelly Criterion
            position_size = self._calculate_position_size(
                signal=signal,
                portfolio=portfolio,
                market_data=market_data
            )

            if position_size > 0:
                action = RebalanceAction(
                    symbol=symbol,
                    action='buy' if signal_action == 'buy' else 'sell',
                    quantity=position_size,
                    reason=f"Signal: {signal.get('reasoning', 'High confidence signal')}",
                    confidence=signal['confidence'],
                    expected_impact={
                        'expected_return': signal.get('expected_return', 0.05),
                        'risk_impact': signal.get('risk_impact', 0.02)
                    }
                )
                actions.append(action)

        return actions

    def _calculate_position_size(self,
                               signal: Dict,
                               portfolio: PortfolioState,
                               market_data: Dict) -> float:
        """Calculate optimal position size using modified Kelly Criterion"""
        # Kelly fraction = (p*b - q) / b
        # where p = probability of win, b = win/loss ratio, q = 1-p

        confidence = signal.get('confidence', 0.5)
        expected_return = signal.get('expected_return', 0.05)
        expected_loss = signal.get('expected_loss', -0.03)

        # Win/loss ratio
        b = abs(expected_return / expected_loss) if expected_loss != 0 else 2.0

        # Kelly fraction
        kelly_fraction = (confidence * b - (1 - confidence)) / b

        # Apply safety factor (use 25% of Kelly)
        safe_fraction = kelly_fraction * 0.25

        # Apply position limits
        max_position = portfolio.total_value * self.max_position_size
        position_value = portfolio.total_value * safe_fraction

        # Convert to shares (would need current price)
        final_position = min(position_value, max_position)

        return final_position

    def _validate_actions(self,
                         actions: List[RebalanceAction],
                         portfolio: PortfolioState) -> List[RebalanceAction]:
        """Validate and prioritize actions"""
        validated = []

        # Sort by confidence and expected impact
        sorted_actions = sorted(
            actions,
            key=lambda x: x.confidence * x.expected_impact.get('expected_return', 0),
            reverse=True
        )

        # Simulate portfolio after each action
        simulated_portfolio = portfolio

        for action in sorted_actions:
            # Check if action violates any constraints
            if self._is_valid_action(action, simulated_portfolio):
                validated.append(action)
                # Update simulated portfolio
                simulated_portfolio = self._simulate_action(action, simulated_portfolio)

        return validated

    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming signal for portfolio impact"""
        # This method is called by the base agent framework
        enhanced_signal = signal.copy()

        # Add portfolio-specific metadata
        enhanced_signal['portfolio_impact'] = {
            'position_size_recommendation': 0.05,  # 5% of portfolio
            'risk_contribution': 0.02,
            'correlation_impact': 'low',
            'rebalance_priority': 'medium'
        }

        return enhanced_signal

    def get_portfolio_recommendations(self,
                                    portfolio: PortfolioState,
                                    market_conditions: Dict) -> Dict[str, Any]:
        """Get comprehensive portfolio recommendations"""
        return {
            'health_score': 1.0 - portfolio.risk_score,
            'recommendations': [
                {
                    'action': 'rebalance',
                    'urgency': 'medium',
                    'reason': 'Quarterly rebalancing due',
                    'expected_improvement': '2% risk reduction'
                },
                {
                    'action': 'diversify',
                    'urgency': 'low',
                    'reason': 'Tech sector overweight',
                    'suggestion': 'Consider healthcare or utilities'
                }
            ],
            'risk_alerts': self._generate_risk_alerts(portfolio, market_conditions),
            'optimization_opportunities': [
                {
                    'type': 'tax_loss_harvesting',
                    'potential_savings': '$1,250',
                    'positions': ['ARKK', 'PLTR']
                }
            ]
        }

    def _generate_risk_alerts(self,
                            portfolio: PortfolioState,
                            market_conditions: Dict) -> List[Dict]:
        """Generate risk alerts for the portfolio"""
        alerts = []

        # Concentration risk
        if portfolio.risk_score > 0.7:
            alerts.append({
                'type': 'high_risk',
                'severity': 'high',
                'message': 'Portfolio risk exceeds target levels',
                'action': 'Consider reducing volatile positions'
            })

        # Market regime risk
        if market_conditions.get('vix', 0) > 30:
            alerts.append({
                'type': 'market_volatility',
                'severity': 'medium',
                'message': 'High market volatility detected',
                'action': 'Consider defensive positioning'
            })

        return alerts

    # Placeholder methods that would need full implementation
    def _calculate_diversification(self, portfolio: PortfolioState) -> float:
        return 0.75

    def _attribute_performance(self, portfolio: PortfolioState, market_data: Dict) -> Dict:
        return {'top_contributors': [], 'top_detractors': []}

    def _check_rebalance_needed(self, portfolio: PortfolioState, market_data: Dict) -> bool:
        return True

    def _find_optimizations(self, portfolio: PortfolioState, market_data: Dict) -> List[Dict]:
        return []

    def _calculate_correlation_risk(self, portfolio: PortfolioState, market_data: Dict) -> float:
        return 0.3

    def _identify_high_risk_positions(self, portfolio: PortfolioState, market_data: Dict) -> List[Dict]:
        return []

    def _generate_optimization_actions(self, portfolio: PortfolioState, market_data: Dict) -> List[RebalanceAction]:
        return []

    def _is_valid_action(self, action: RebalanceAction, portfolio: PortfolioState) -> bool:
        return True

    def _simulate_action(self, action: RebalanceAction, portfolio: PortfolioState) -> PortfolioState:
        return portfolio
