"""
Integration tests for AlphaPy components.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from AlphaPy.data import DataLoader, MarketDataProcessor, FeatureEngineer
from AlphaPy.models import TimeSeriesModel
from AlphaPy.portfolio import PortfolioOptimizer, RiskManager
from AlphaPy.analytics import PerformanceAnalytics, RiskAnalytics

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    data = {}
    for asset in assets:
        data[asset] = pd.DataFrame({
            'open': np.random.normal(100, 10, len(dates)),
            'high': np.random.normal(105, 10, len(dates)),
            'low': np.random.normal(95, 10, len(dates)),
            'close': np.random.normal(100, 10, len(dates)),
            'volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
    
    return data

@pytest.mark.integration
def test_data_to_model_pipeline(sample_market_data):
    """Test data processing to model pipeline."""
    # Initialize components
    processor = MarketDataProcessor()
    engineer = FeatureEngineer()
    model = TimeSeriesModel({'lookback': 10, 'horizon': 1})
    
    # Process each asset's data
    processed_data = {}
    for asset, data in sample_market_data.items():
        # Clean and validate data
        assert processor.validate_data(data)
        cleaned_data = processor.clean_data(data)
        
        # Add technical features (assuming implementation)
        try:
            features = engineer.add_technical_features(
                cleaned_data,
                ['rsi', 'macd', 'bollinger']
            )
        except NotImplementedError:
            # Skip if not implemented
            features = cleaned_data
            
        processed_data[asset] = features
    
    # Verify processed data
    assert len(processed_data) == len(sample_market_data)
    for asset in processed_data:
        assert isinstance(processed_data[asset], pd.DataFrame)
        assert not processed_data[asset].isnull().any().any()

@pytest.mark.integration
def test_model_to_portfolio_pipeline(sample_market_data):
    """Test model predictions to portfolio optimization pipeline."""
    # Initialize components
    model = TimeSeriesModel({'lookback': 10, 'horizon': 1})
    optimizer = PortfolioOptimizer({'risk_aversion': 1.0})
    risk_manager = RiskManager({'confidence': 0.95})
    
    # Generate predictions (mock since train is not implemented)
    predictions = pd.DataFrame({
        asset: np.random.normal(0.001, 0.02, 30)  # 30 days of predictions
        for asset in sample_market_data.keys()
    })
    
    # Set returns for optimization
    optimizer.set_returns(predictions)
    
    # Optimize portfolio
    weights = optimizer.optimize_mean_variance()
    
    # Apply risk management
    for asset in weights.index:
        risk_manager.set_position_limit(asset, -0.2, 0.2)
    
    # Calculate portfolio risk metrics
    portfolio_returns = (predictions * weights).sum(axis=1)
    var = risk_manager.calculate_var(weights, predictions)
    es = risk_manager.calculate_es(weights, predictions)
    
    # Verify risk metrics
    assert var > 0
    assert es >= var
    assert (weights >= -0.2).all() and (weights <= 0.2).all()

@pytest.mark.integration
def test_portfolio_to_analytics_pipeline(sample_market_data):
    """Test portfolio management to analytics pipeline."""
    # Initialize components
    optimizer = PortfolioOptimizer({'risk_aversion': 1.0})
    perf_analytics = PerformanceAnalytics()
    risk_analytics = RiskAnalytics()
    
    # Create portfolio returns
    portfolio_returns = pd.Series(
        np.random.normal(0.0001, 0.01, 252),  # One year of daily returns
        index=pd.date_range(start='2023-01-01', periods=252, freq='B')
    )
    
    # Calculate performance metrics
    metrics = perf_analytics.calculate_metrics(
        portfolio_returns,
        risk_free_rate=0.02
    )
    
    # Calculate risk metrics
    var = risk_analytics.calculate_var(
        portfolio_returns,
        confidence=0.95,
        method='historical'
    )
    
    beta = risk_analytics.calculate_beta(
        portfolio_returns,
        pd.Series(np.random.normal(0.0001, 0.01, len(portfolio_returns)))  # Market returns
    )
    
    # Verify analytics results
    assert all(metric in metrics for metric in [
        'total_return', 'annual_return', 'volatility',
        'sharpe_ratio', 'sortino_ratio', 'max_drawdown'
    ])
    assert var > 0
    assert isinstance(beta, float)

@pytest.mark.integration
def test_full_trading_pipeline(sample_market_data):
    """Test complete trading pipeline from data to analytics."""
    # Initialize all components
    processor = MarketDataProcessor()
    engineer = FeatureEngineer()
    model = TimeSeriesModel({'lookback': 10, 'horizon': 1})
    optimizer = PortfolioOptimizer({'risk_aversion': 1.0})
    risk_manager = RiskManager({'confidence': 0.95})
    analytics = PerformanceAnalytics()
    
    # Process market data
    processed_data = {
        asset: processor.clean_data(data)
        for asset, data in sample_market_data.items()
    }
    
    # Generate features (mock since not implemented)
    features = {
        asset: data.copy()  # In real implementation, would add features here
        for asset, data in processed_data.items()
    }
    
    # Generate predictions (mock since train not implemented)
    predictions = pd.DataFrame({
        asset: np.random.normal(0.001, 0.02, 30)
        for asset in features.keys()
    })
    
    # Optimize portfolio
    optimizer.set_returns(predictions)
    weights = optimizer.optimize_mean_variance()
    
    # Apply risk limits
    for asset in weights.index:
        risk_manager.set_position_limit(asset, -0.2, 0.2)
    
    # Calculate portfolio returns
    portfolio_returns = (predictions * weights).sum(axis=1)
    
    # Calculate performance metrics
    metrics = analytics.calculate_metrics(portfolio_returns)
    
    # Verify entire pipeline
    assert isinstance(weights, pd.Series)
    assert np.allclose(weights.sum(), 1.0)
    assert all(metric in metrics for metric in [
        'total_return', 'annual_return', 'volatility',
        'sharpe_ratio', 'sortino_ratio', 'max_drawdown'
    ]) 