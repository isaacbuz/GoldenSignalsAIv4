"""
Property-based tests for AlphaPy components.
"""

import pytest
from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from AlphaPy.portfolio import PortfolioOptimizer, RiskManager
from AlphaPy.analytics import PerformanceAnalytics, RiskAnalytics

# Custom strategies
@st.composite
def returns_series(draw):
    """Generate random return series."""
    length = draw(st.integers(min_value=10, max_value=1000))
    returns = draw(st.lists(
        st.floats(min_value=-0.5, max_value=0.5),
        min_size=length,
        max_size=length
    ))
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    return pd.Series(returns, index=dates)

@st.composite
def portfolio_weights(draw):
    """Generate random portfolio weights that sum to 1."""
    n_assets = draw(st.integers(min_value=2, max_value=50))
    weights = draw(st.lists(
        st.floats(min_value=0, max_value=1),
        min_size=n_assets,
        max_size=n_assets
    ))
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize to sum to 1
    return pd.Series(weights, index=[f'ASSET_{i}' for i in range(n_assets)])

@pytest.mark.property
@given(returns=returns_series())
def test_performance_metrics_properties(returns):
    """Test properties of performance metrics."""
    analytics = PerformanceAnalytics()
    metrics = analytics.calculate_metrics(returns)
    
    # Volatility should be non-negative
    assert metrics['volatility'] >= 0
    
    # Max drawdown should be non-positive
    assert metrics['max_drawdown'] <= 0
    
    # Total return relationship
    total_return = metrics['total_return']
    compound_return = (1 + returns).prod() - 1
    assert np.allclose(total_return, compound_return, rtol=1e-10)
    
    # Sharpe ratio relationship with returns and volatility
    if metrics['volatility'] > 0:
        expected_sharpe = (returns.mean() * 252) / (metrics['volatility'])
        assert np.allclose(metrics['sharpe_ratio'], expected_sharpe, rtol=1e-10)

@pytest.mark.property
@given(
    returns=returns_series(),
    confidence=st.floats(min_value=0.9, max_value=0.99)
)
def test_risk_metrics_properties(returns, confidence):
    """Test properties of risk metrics."""
    risk_analytics = RiskAnalytics()
    
    # Calculate VaR and ES
    var = risk_analytics.calculate_var(returns, confidence=confidence)
    es = risk_analytics.calculate_es(returns, confidence=confidence)
    
    # VaR should be positive
    assert var > 0
    
    # ES should be greater than or equal to VaR
    assert es >= var
    
    # VaR should be at the correct percentile
    empirical_var = np.percentile(returns, (1 - confidence) * 100)
    assert np.allclose(var, -empirical_var, rtol=1e-10)
    
    # ES should be the average of tail events
    tail_returns = returns[returns <= -var]
    if len(tail_returns) > 0:
        empirical_es = -tail_returns.mean()
        assert np.allclose(es, empirical_es, rtol=1e-10)

@pytest.mark.property
@given(weights=portfolio_weights())
def test_portfolio_optimization_properties(weights):
    """Test properties of portfolio optimization."""
    n_assets = len(weights)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    
    # Generate returns with some correlation structure
    cov_matrix = np.random.normal(0, 0.1, (n_assets, n_assets))
    cov_matrix = cov_matrix.T @ cov_matrix  # Ensure positive semi-definite
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            mean=np.random.normal(0.0001, 0.0001, n_assets),
            cov=cov_matrix,
            size=len(dates)
        ),
        index=dates,
        columns=weights.index
    )
    
    optimizer = PortfolioOptimizer({'risk_aversion': 1.0})
    optimizer.set_returns(returns)
    
    # Optimize portfolio
    optimal_weights = optimizer.optimize_mean_variance()
    
    # Weights should sum to 1
    assert np.allclose(optimal_weights.sum(), 1.0, rtol=1e-10)
    
    # All weights should be between 0 and 1 (long-only constraint)
    assert (optimal_weights >= 0).all()
    assert (optimal_weights <= 1).all()
    
    # Portfolio return and risk
    port_return = (returns @ optimal_weights).mean()
    port_risk = np.sqrt((optimal_weights @ returns.cov() @ optimal_weights))
    
    # Higher risk aversion should lead to lower risk
    optimizer_conservative = PortfolioOptimizer({'risk_aversion': 2.0})
    optimizer_conservative.set_returns(returns)
    conservative_weights = optimizer_conservative.optimize_mean_variance()
    conservative_risk = np.sqrt(
        (conservative_weights @ returns.cov() @ conservative_weights)
    )
    
    assert conservative_risk <= port_risk * 1.1  # Allow for small numerical differences

@pytest.mark.property
@given(
    returns=returns_series(),
    window=st.integers(min_value=5, max_value=100)
)
def test_rolling_metrics_properties(returns, window):
    """Test properties of rolling risk metrics."""
    risk_analytics = RiskAnalytics()
    
    # Calculate rolling VaR
    rolling_var = pd.Series([
        risk_analytics.calculate_var(returns[i:i+window])
        for i in range(len(returns) - window)
    ])
    
    # Calculate rolling ES
    rolling_es = pd.Series([
        risk_analytics.calculate_es(returns[i:i+window])
        for i in range(len(returns) - window)
    ])
    
    # All VaR values should be positive
    assert (rolling_var > 0).all()
    
    # ES should always be greater than or equal to VaR
    assert (rolling_es >= rolling_var).all()
    
    # Metrics should be more stable for larger windows
    if len(rolling_var) > 2:
        small_window_var = pd.Series([
            risk_analytics.calculate_var(returns[i:i+5])
            for i in range(len(returns) - 5)
        ])
        
        assert rolling_var.std() <= small_window_var.std() 