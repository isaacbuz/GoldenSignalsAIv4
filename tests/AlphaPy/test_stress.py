"""
Stress tests for AlphaPy components.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from AlphaPy.data import MarketDataProcessor
from AlphaPy.models import TimeSeriesModel
from AlphaPy.portfolio import PortfolioOptimizer, RiskManager
from AlphaPy.analytics import PerformanceAnalytics, RiskAnalytics

@pytest.mark.stress
def test_large_data_processing():
    """Test processing of large datasets."""
    # Generate large dataset (1 year of minute data for 100 assets)
    dates = pd.date_range(
        start='2023-01-01',
        end='2023-12-31',
        freq='T'  # Minute frequency
    )
    assets = [f'ASSET_{i}' for i in range(100)]
    
    data = {}
    for asset in assets:
        data[asset] = pd.DataFrame({
            'open': np.random.normal(100, 10, len(dates)),
            'high': np.random.normal(105, 10, len(dates)),
            'low': np.random.normal(95, 10, len(dates)),
            'close': np.random.normal(100, 10, len(dates)),
            'volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
    
    processor = MarketDataProcessor()
    
    # Process each asset's data
    start_time = datetime.now()
    for asset, df in data.items():
        cleaned_data = processor.clean_data(df)
        assert not cleaned_data.isnull().any().any()
    
    processing_time = (datetime.now() - start_time).total_seconds()
    assert processing_time < 300  # Should process within 5 minutes

@pytest.mark.stress
def test_portfolio_optimization_stress():
    """Test portfolio optimization with large number of assets."""
    # Generate returns for 500 assets
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_assets = 500
    
    returns = pd.DataFrame(
        np.random.normal(0.0001, 0.02, (len(dates), n_assets)),
        index=dates,
        columns=[f'ASSET_{i}' for i in range(n_assets)]
    )
    
    optimizer = PortfolioOptimizer({'risk_aversion': 1.0})
    optimizer.set_returns(returns)
    
    # Test optimization performance
    start_time = datetime.now()
    weights = optimizer.optimize_mean_variance()
    optimization_time = (datetime.now() - start_time).total_seconds()
    
    assert len(weights) == n_assets
    assert np.allclose(weights.sum(), 1.0)
    assert optimization_time < 300  # Should optimize within 5 minutes

@pytest.mark.stress
def test_risk_calculation_stress():
    """Test risk calculations with high-frequency data."""
    # Generate minute-level returns for a year
    dates = pd.date_range(
        start='2023-01-01',
        end='2023-12-31',
        freq='T'  # Minute frequency
    )
    
    returns = pd.Series(
        np.random.normal(0.0001/1440, 0.01/np.sqrt(1440), len(dates)),
        index=dates
    )
    
    risk_analytics = RiskAnalytics()
    
    # Calculate VaR and ES for different lookback windows
    windows = [1000, 5000, 10000, 50000]
    
    for window in windows:
        start_time = datetime.now()
        
        # Rolling VaR calculation
        for i in range(window, len(returns), 1000):
            window_returns = returns[i-window:i]
            var = risk_analytics.calculate_var(
                window_returns,
                confidence=0.99,
                method='historical'
            )
            es = risk_analytics.calculate_es(
                window_returns,
                confidence=0.99
            )
            
            assert var > 0
            assert es >= var
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        assert calculation_time < 300  # Should calculate within 5 minutes

@pytest.mark.stress
def test_concurrent_portfolio_updates():
    """Test concurrent portfolio updates and risk calculations."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    # Shared portfolio and risk manager
    optimizer = PortfolioOptimizer({'risk_aversion': 1.0})
    risk_manager = RiskManager({'confidence': 0.95})
    
    # Shared data structure for positions
    positions = {}
    positions_lock = threading.Lock()
    
    def update_position(asset):
        """Update position for an asset."""
        with positions_lock:
            current_position = positions.get(asset, 0)
            # Simulate position update
            new_position = current_position + np.random.normal(0, 100)
            positions[asset] = new_position
            
            # Calculate risk metrics
            portfolio_positions = pd.Series(positions)
            var = risk_manager.calculate_var(
                portfolio_positions,
                pd.DataFrame(np.random.normal(0, 0.02, (100, len(positions))))
            )
            return var
    
    # Initialize with 100 assets
    assets = [f'ASSET_{i}' for i in range(100)]
    
    # Run concurrent updates
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(update_position, assets * 10))  # 1000 updates
    
    assert len(positions) == 100
    assert all(isinstance(var, float) for var in results)

@pytest.mark.stress
def test_model_memory_usage():
    """Test memory usage of time series model with large sequences."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Generate large dataset
    n_sequences = 10000
    sequence_length = 1000
    n_features = 50
    
    X = np.random.normal(0, 1, (n_sequences, sequence_length, n_features))
    y = np.random.normal(0, 1, (n_sequences, 1))
    
    model = TimeSeriesModel({
        'lookback': sequence_length,
        'horizon': 1
    })
    
    # Test memory usage during sequence preparation
    sequences, targets = model.prepare_sequences(
        pd.DataFrame(X.reshape(-1, n_features))
    )
    
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = current_memory - initial_memory
    
    # Memory increase should be reasonable (less than 4GB)
    assert memory_increase < 4000  # MB 