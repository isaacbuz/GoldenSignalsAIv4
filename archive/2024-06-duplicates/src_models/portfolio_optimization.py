"""
Portfolio Optimization Models for GoldenSignalsAI
Modern portfolio theory with ML enhancements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import cvxpy as cp
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PortfolioWeights:
    """Portfolio allocation results"""
    weights: Dict[str, float]
    expected_return: float
    risk: float
    sharpe_ratio: float
    metadata: Dict[str, any]


class ModernPortfolioOptimizer:
    """Modern Portfolio Theory with enhancements"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.returns = None
        self.cov_matrix = None
    
    def fit(self, returns: pd.DataFrame):
        """Calculate expected returns and covariance"""
        self.returns = returns
        self.expected_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.assets = returns.columns.tolist()
    
    def optimize_sharpe(self) -> PortfolioWeights:
        """Maximize Sharpe ratio"""
        n_assets = len(self.assets)
        
        def neg_sharpe(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_std
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(neg_sharpe, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        weights = result.x
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        return PortfolioWeights(
            weights=dict(zip(self.assets, weights)),
            expected_return=portfolio_return,
            risk=portfolio_risk,
            sharpe_ratio=sharpe,
            metadata={'optimization_success': result.success}
        )
    
    def optimize_min_variance(self) -> PortfolioWeights:
        """Minimize portfolio variance"""
        n_assets = len(self.assets)
        
        # Use cvxpy for convex optimization
        weights = cp.Variable(n_assets)
        risk = cp.quad_form(weights, self.cov_matrix.values)
        
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        problem = cp.Problem(cp.Minimize(risk), constraints)
        problem.solve()
        
        w = weights.value
        portfolio_return = np.dot(w, self.expected_returns)
        portfolio_risk = np.sqrt(risk.value)
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        return PortfolioWeights(
            weights=dict(zip(self.assets, w)),
            expected_return=portfolio_return,
            risk=portfolio_risk,
            sharpe_ratio=sharpe,
            metadata={'solver_status': problem.status}
        )


class RiskParityOptimizer:
    """Risk Parity Portfolio Optimization"""
    
    def __init__(self):
        self.cov_matrix = None
        self.assets = None
    
    def fit(self, returns: pd.DataFrame):
        """Calculate covariance matrix"""
        self.cov_matrix = returns.cov()
        self.assets = returns.columns.tolist()
    
    def optimize(self) -> PortfolioWeights:
        """Equal risk contribution portfolio"""
        n_assets = len(self.assets)
        
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
        
        def objective(weights):
            contrib = risk_contribution(weights)
            # Minimize deviation from equal contribution
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib)**2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        weights = result.x
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        return PortfolioWeights(
            weights=dict(zip(self.assets, weights)),
            expected_return=None,  # Not optimizing for return
            risk=portfolio_risk,
            sharpe_ratio=None,
            metadata={'risk_contributions': risk_contribution(weights).tolist()}
        )


class CVaROptimizer:
    """Conditional Value at Risk Optimization"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.returns = None
        self.assets = None
    
    def fit(self, returns: pd.DataFrame):
        """Store historical returns"""
        self.returns = returns
        self.assets = returns.columns.tolist()
    
    def optimize(self, target_return: Optional[float] = None) -> PortfolioWeights:
        """Minimize CVaR"""
        n_assets = len(self.assets)
        n_scenarios = len(self.returns)
        
        # Decision variables
        weights = cp.Variable(n_assets)
        z = cp.Variable()  # VaR
        u = cp.Variable(n_scenarios)  # Auxiliary variables
        
        # Portfolio returns for each scenario
        portfolio_returns = self.returns.values @ weights
        
        # CVaR formulation
        alpha = 1 - self.confidence_level
        cvar = z + (1/(n_scenarios * alpha)) * cp.sum(u)
        
        constraints = [
            u >= 0,
            u >= -portfolio_returns - z,
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        # Add return constraint if specified
        if target_return is not None:
            expected_return = self.returns.mean().values @ weights
            constraints.append(expected_return >= target_return)
        
        problem = cp.Problem(cp.Minimize(cvar), constraints)
        problem.solve()
        
        w = weights.value
        expected_return = np.dot(w, self.returns.mean())
        
        return PortfolioWeights(
            weights=dict(zip(self.assets, w)),
            expected_return=expected_return,
            risk=cvar.value,
            sharpe_ratio=None,
            metadata={
                'VaR': z.value,
                'CVaR': cvar.value,
                'confidence_level': self.confidence_level
            }
        )


class BlackLittermanOptimizer:
    """Black-Litterman Model Implementation"""
    
    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.market_caps = None
        self.cov_matrix = None
    
    def fit(self, returns: pd.DataFrame, market_caps: pd.Series):
        """Initialize with market data"""
        self.cov_matrix = returns.cov()
        self.market_caps = market_caps
        self.assets = returns.columns.tolist()
        
        # Calculate equilibrium returns
        self.market_weights = market_caps / market_caps.sum()
        self.equilibrium_returns = self.risk_aversion * self.cov_matrix @ self.market_weights
    
    def optimize(self, views: pd.DataFrame, view_confidence: np.ndarray) -> PortfolioWeights:
        """
        Optimize with views
        views: DataFrame with columns for each asset, rows for each view
        view_confidence: Array of confidence levels for each view
        """
        P = views.values  # Pick matrix
        Q = views.sum(axis=1).values  # View returns
        omega = np.diag(view_confidence)  # Uncertainty matrix
        
        # Black-Litterman formula
        tau_sigma_inv = np.linalg.inv(self.tau * self.cov_matrix)
        
        # Posterior covariance
        posterior_cov = np.linalg.inv(
            tau_sigma_inv + P.T @ np.linalg.inv(omega) @ P
        )
        
        # Posterior returns
        posterior_returns = posterior_cov @ (
            tau_sigma_inv @ self.equilibrium_returns + 
            P.T @ np.linalg.inv(omega) @ Q
        )
        
        # Optimize using posterior
        optimizer = ModernPortfolioOptimizer(risk_free_rate=0.02)
        optimizer.expected_returns = pd.Series(posterior_returns, index=self.assets)
        optimizer.cov_matrix = pd.DataFrame(posterior_cov, 
                                          index=self.assets, 
                                          columns=self.assets)
        optimizer.assets = self.assets
        
        result = optimizer.optimize_sharpe()
        result.metadata['posterior_returns'] = dict(zip(self.assets, posterior_returns))
        
        return result


class KellyOptimizer:
    """Kelly Criterion for Optimal Position Sizing"""
    
    def __init__(self, max_leverage: float = 1.0):
        self.max_leverage = max_leverage
    
    def calculate_kelly_fraction(self, 
                                win_prob: float, 
                                win_return: float, 
                                loss_return: float) -> float:
        """
        Calculate Kelly fraction for binary outcome
        win_prob: Probability of winning
        win_return: Return if win (e.g., 0.1 for 10%)
        loss_return: Return if loss (e.g., -0.05 for -5%)
        """
        if loss_return >= 0:
            return 0  # No edge
        
        q = 1 - win_prob
        b = win_return / abs(loss_return)
        
        kelly = (win_prob * b - q) / b
        
        # Apply leverage constraint
        return min(max(kelly, 0), self.max_leverage)
    
    def optimize_multi_asset(self, 
                           expected_returns: pd.Series,
                           cov_matrix: pd.DataFrame,
                           confidence: float = 0.25) -> PortfolioWeights:
        """
        Multi-asset Kelly optimization
        confidence: Scaling factor (fractional Kelly)
        """
        # Kelly weights = (Σ^-1) * μ / λ
        # where λ is chosen to satisfy leverage constraint
        
        inv_cov = np.linalg.inv(cov_matrix.values)
        raw_kelly = inv_cov @ expected_returns.values
        
        # Scale to meet leverage constraint
        total_position = np.sum(np.abs(raw_kelly))
        if total_position > self.max_leverage:
            raw_kelly = raw_kelly * self.max_leverage / total_position
        
        # Apply confidence scaling (fractional Kelly)
        kelly_weights = raw_kelly * confidence
        
        # Calculate metrics
        portfolio_return = np.dot(kelly_weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(kelly_weights.T, np.dot(cov_matrix, kelly_weights)))
        
        return PortfolioWeights(
            weights=dict(zip(expected_returns.index, kelly_weights)),
            expected_return=portfolio_return,
            risk=portfolio_risk,
            sharpe_ratio=portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
            metadata={
                'total_leverage': np.sum(np.abs(kelly_weights)),
                'confidence_factor': confidence
            }
        )


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_assets = 5
    n_days = 252
    
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.0002] * n_assets,
            cov=np.eye(n_assets) * 0.01,
            size=n_days
        ),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Modern Portfolio Theory
    mpt = ModernPortfolioOptimizer()
    mpt.fit(returns)
    sharpe_portfolio = mpt.optimize_sharpe()
    print("Sharpe Optimal Portfolio:")
    print(sharpe_portfolio.weights)
    
    # Risk Parity
    rp = RiskParityOptimizer()
    rp.fit(returns)
    rp_portfolio = rp.optimize()
    print("\nRisk Parity Portfolio:")
    print(rp_portfolio.weights)
    
    # CVaR Optimization
    cvar = CVaROptimizer(confidence_level=0.95)
    cvar.fit(returns)
    cvar_portfolio = cvar.optimize()
    print("\nCVaR Optimal Portfolio:")
    print(cvar_portfolio.weights) 