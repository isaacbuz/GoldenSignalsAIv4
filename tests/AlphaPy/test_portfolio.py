"""
Tests for AlphaPy portfolio management functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from AlphaPy.portfolio import (
    PortfolioOptimizer,
    RiskManager,
    PositionSizer
)

@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    returns = pd.DataFrame(
        np.random.normal(0.0001, 0.02, (len(dates), len(assets))),
        index=dates,
        columns=assets
    )
    return returns

def test_portfolio_optimizer(sample_returns):
    """Test portfolio optimization functionality."""
    optimizer = PortfolioOptimizer({'risk_aversion': 1.0})
    
    # Set returns data
    optimizer.set_returns(sample_returns)
    
    # Test mean-variance optimization
    weights = optimizer.optimize_mean_variance()
    
    # Verify constraints
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(sample_returns.columns)
    assert np.allclose(weights.sum(), 1.0)  # weights sum to 1
    assert (weights >= 0).all()  # long-only constraint

def test_risk_manager(sample_returns):
    """Test risk management functionality."""
    risk_manager = RiskManager({'confidence': 0.95})
    
    # Test position limits
    asset = 'AAPL'
    risk_manager.set_position_limit(asset, -0.1, 0.2)
    assert asset in risk_manager.position_limits
    assert risk_manager.position_limits[asset]['min'] == -0.1
    assert risk_manager.position_limits[asset]['max'] == 0.2
    
    # Test VaR calculation
    portfolio = pd.Series([0.25] * 4, index=sample_returns.columns)
    var = risk_manager.calculate_var(portfolio, sample_returns)
    assert var > 0  # VaR should be positive
    
    # Test Expected Shortfall calculation
    es = risk_manager.calculate_es(portfolio, sample_returns)
    assert es > var  # ES should be greater than VaR

def test_position_sizer():
    """Test position sizing functionality."""
    config = {
        'portfolio_value': 1000000,
        'max_position_size': 0.1
    }
    sizer = PositionSizer(config)
    
    # Test basic position sizing
    signal = 0.5
    price = 100
    volatility = 0.02
    
    position = sizer.calculate_position_size(signal, price, volatility)
    
    # Verify position constraints
    max_position_value = config['portfolio_value'] * config['max_position_size']
    assert abs(position * price) <= max_position_value
    
    # Test position adjustment
    current_position = 1000
    adjusted_position = sizer.adjust_for_limits(
        position_size=1500,
        current_position=current_position,
        price=price
    )
    
    # Verify position change limits
    max_change = int(abs(current_position) * 0.2)  # 20% change limit
    assert abs(adjusted_position - current_position) <= max_change

@pytest.mark.parametrize("signal,expected_sign", [
    (1.0, 1),
    (-1.0, -1),
    (0.5, 1),
    (-0.5, -1),
    (0.0, 0)
])
def test_position_sizer_signal_direction(signal, expected_sign):
    """Test position sizing with different signals."""
    sizer = PositionSizer({
        'portfolio_value': 1000000,
        'max_position_size': 0.1
    })
    
    position = sizer.calculate_position_size(
        signal=signal,
        price=100,
        volatility=0.02
    )
    
    if expected_sign == 0:
        assert position == 0
    else:
        assert np.sign(position) == expected_sign 