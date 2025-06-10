"""
Tests for AlphaPy data management functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from AlphaPy.data import DataLoader, FeatureEngineer, MarketDataProcessor

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.normal(1000000, 200000, len(dates))
    }, index=dates)
    return data

def test_market_data_processor_validation(sample_market_data):
    """Test market data validation."""
    processor = MarketDataProcessor()
    
    # Test valid data
    assert processor.validate_data(sample_market_data) == True
    
    # Test invalid data (missing columns)
    invalid_data = sample_market_data.drop(['volume'], axis=1)
    assert processor.validate_data(invalid_data) == False
    
    # Test empty data
    empty_data = pd.DataFrame()
    assert processor.validate_data(empty_data) == False

def test_market_data_processor_cleaning(sample_market_data):
    """Test market data cleaning."""
    processor = MarketDataProcessor()
    
    # Add some missing values
    data_with_missing = sample_market_data.copy()
    data_with_missing.iloc[1:3] = np.nan
    
    cleaned_data = processor.clean_data(data_with_missing)
    
    # Check that missing values were filled
    assert not cleaned_data.isnull().any().any()
    
    # Check data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        assert pd.api.types.is_numeric_dtype(cleaned_data[col])

def test_market_data_processor_resampling(sample_market_data):
    """Test market data resampling."""
    processor = MarketDataProcessor()
    
    # Test resampling to 2-day frequency
    resampled = processor.resample_data(sample_market_data, '2D')
    
    # Check that length is approximately halved
    assert len(resampled) == len(sample_market_data) // 2 + len(sample_market_data) % 2
    
    # Check that OHLCV values are properly aggregated
    assert (resampled['high'] >= resampled['open']).all()
    assert (resampled['high'] >= resampled['close']).all()
    assert (resampled['low'] <= resampled['open']).all()
    assert (resampled['low'] <= resampled['close']).all()

def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization."""
    engineer = FeatureEngineer()
    assert isinstance(engineer.features, dict)

def test_data_loader_initialization():
    """Test DataLoader initialization."""
    config = {'source': 'yahoo'}
    loader = DataLoader(config)
    assert loader.config == config 