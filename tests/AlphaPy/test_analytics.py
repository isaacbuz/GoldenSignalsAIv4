"""
Tests for AlphaPy analytics functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from AlphaPy.analytics import (
    PerformanceAnalytics,
    RiskAnalytics,
    AttributionAnalysis
)

@pytest.fixture
def sample_portfolio_data():
    """Create sample portfolio data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create returns data
    returns = pd.Series(
        np.random.normal(0.0001, 0.01, len(dates)),
        index=dates
    )
    
    # Create price data
    prices = (1 + returns).cumprod() * 100
    
    return prices, returns

@pytest.fixture
def sample_factor_data():
    """Create sample factor data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    factors = pd.DataFrame({
        'market': np.random.normal(0.0001, 0.01, len(dates)),
        'size': np.random.normal(0, 0.005, len(dates)),
        'value': np.random.normal(0, 0.005, len(dates))
    }, index=dates)
    
    return factors

def test_performance_analytics_returns(sample_portfolio_data):
    """Test return calculations."""
    prices, _ = sample_portfolio_data
    analytics = PerformanceAnalytics()
    
    # Test arithmetic returns
    arith_returns = analytics.calculate_returns(prices, method='arithmetic')
    assert isinstance(arith_returns, pd.Series)
    assert len(arith_returns) == len(prices) - 1
    
    # Test log returns
    log_returns = analytics.calculate_returns(prices, method='log')
    assert isinstance(log_returns, pd.Series)
    assert len(log_returns) == len(prices) - 1
    
    # Test invalid method
    with pytest.raises(ValueError):
        analytics.calculate_returns(prices, method='invalid')

def test_performance_analytics_metrics(sample_portfolio_data):
    """Test performance metrics calculations."""
    _, returns = sample_portfolio_data
    analytics = PerformanceAnalytics()
    
    metrics = analytics.calculate_metrics(returns, risk_free_rate=0.02)
    
    # Check required metrics
    required_metrics = [
        'total_return',
        'annual_return',
        'volatility',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown'
    ]
    
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
    
    # Verify metric constraints
    assert metrics['volatility'] >= 0
    assert metrics['max_drawdown'] <= 0

def test_risk_analytics_var(sample_portfolio_data):
    """Test Value at Risk calculations."""
    _, returns = sample_portfolio_data
    analytics = RiskAnalytics()
    
    # Test historical VaR
    hist_var = analytics.calculate_var(returns, confidence=0.95, method='historical')
    assert hist_var > 0
    
    # Test parametric VaR
    param_var = analytics.calculate_var(returns, confidence=0.95, method='parametric')
    assert param_var > 0
    
    # Test invalid method
    with pytest.raises(ValueError):
        analytics.calculate_var(returns, method='invalid')

def test_risk_analytics_es(sample_portfolio_data):
    """Test Expected Shortfall calculations."""
    _, returns = sample_portfolio_data
    analytics = RiskAnalytics()
    
    # Calculate ES
    es = analytics.calculate_es(returns, confidence=0.95)
    var = analytics.calculate_var(returns, confidence=0.95)
    
    assert es > 0
    assert es >= var  # ES should be greater than or equal to VaR

def test_risk_analytics_beta(sample_portfolio_data, sample_factor_data):
    """Test beta calculations."""
    _, returns = sample_portfolio_data
    analytics = RiskAnalytics()
    
    # Calculate beta to market factor
    beta = analytics.calculate_beta(returns, sample_factor_data['market'])
    
    assert isinstance(beta, float)

def test_attribution_analysis_factors(sample_portfolio_data, sample_factor_data):
    """Test factor attribution analysis."""
    _, returns = sample_portfolio_data
    analytics = AttributionAnalysis()
    
    # Perform factor attribution
    attribution = analytics.factor_attribution(returns, sample_factor_data)
    
    # Check required components
    assert 'factor_contribution' in attribution
    assert 'alpha' in attribution
    assert 'r_squared' in attribution
    
    # Verify attribution components
    assert isinstance(attribution['factor_contribution'], pd.Series)
    assert len(attribution['factor_contribution']) == len(sample_factor_data.columns)
    assert 0 <= attribution['r_squared'] <= 1

def test_attribution_analysis_sectors(sample_portfolio_data):
    """Test sector attribution analysis."""
    _, returns = sample_portfolio_data
    analytics = AttributionAnalysis()
    
    # Create sample sector data
    sectors = ['Tech', 'Finance', 'Healthcare']
    sector_weights = pd.DataFrame({
        sector: np.random.uniform(0, 1, len(returns)) for sector in sectors
    }, index=returns.index)
    sector_weights = sector_weights.div(sector_weights.sum(axis=1), axis=0)
    
    sector_returns = pd.DataFrame({
        sector: np.random.normal(0.0001, 0.01, len(returns)) for sector in sectors
    }, index=returns.index)
    
    # Perform sector attribution
    attribution = analytics.sector_attribution(
        returns,
        sector_weights,
        sector_returns
    )
    
    # Check attribution components
    required_columns = ['contribution', 'allocation', 'selection', 'total_effect']
    assert all(col in attribution.columns for col in required_columns)
    assert len(attribution) == len(sectors) 