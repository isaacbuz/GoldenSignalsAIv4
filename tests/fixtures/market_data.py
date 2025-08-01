"""Market data fixtures for testing"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')

    # Generate realistic price movements
    np.random.seed(42)
    base_price = 100
    prices = []

    for i in range(100):
        change = np.random.normal(0, 0.02)
        base_price *= (1 + change)
        prices.append(base_price)

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': [p * 1.002 for p in prices],
        'volume': np.random.randint(1000, 10000, 100)
    })

    return data

@pytest.fixture
def sample_market_data_dict():
    """Generate market data in dictionary format"""
    return {
        "symbol": "AAPL",
        "close": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                 111, 110, 112, 114, 113, 115, 117, 116, 118, 120],
        "open": [99, 101, 100, 102, 104, 103, 105, 107, 106, 108,
                110, 109, 111, 113, 112, 114, 116, 115, 117, 119],
        "high": [101, 103, 102, 104, 106, 105, 107, 109, 108, 110,
                112, 111, 113, 115, 114, 116, 118, 117, 119, 121],
        "low": [98, 100, 99, 101, 103, 102, 104, 106, 105, 107,
               109, 108, 110, 112, 111, 113, 115, 114, 116, 118],
        "volume": [1000000, 1100000, 900000, 1200000, 1300000,
                  800000, 1400000, 1500000, 1000000, 1600000,
                  1700000, 1100000, 1800000, 1900000, 1200000,
                  2000000, 2100000, 1300000, 2200000, 2300000],
        "timestamp": pd.date_range(end=datetime.now(), periods=20, freq='H').tolist()
    }

@pytest.fixture
def bullish_market_data():
    """Generate bullish trending market data"""
    prices = []
    base = 100
    for i in range(50):
        base += np.random.uniform(0, 2)
        prices.append(base)

    return {
        "close": prices,
        "volume": [1000000 + i * 10000 for i in range(50)]
    }

@pytest.fixture
def bearish_market_data():
    """Generate bearish trending market data"""
    prices = []
    base = 100
    for i in range(50):
        base -= np.random.uniform(0, 1.5)
        prices.append(base)

    return {
        "close": prices,
        "volume": [1000000 - i * 5000 for i in range(50)]
    }

@pytest.fixture
def sideways_market_data():
    """Generate sideways/ranging market data"""
    prices = []
    base = 100
    for i in range(50):
        base += np.random.uniform(-1, 1)
        prices.append(base)

    return {
        "close": prices,
        "volume": [1000000 + np.random.randint(-50000, 50000) for _ in range(50)]
    }

@pytest.fixture
def high_volatility_data():
    """Generate high volatility market data"""
    prices = []
    base = 100
    for i in range(50):
        base *= (1 + np.random.uniform(-0.05, 0.05))
        prices.append(base)

    return {
        "close": prices,
        "volume": [np.random.randint(500000, 2000000) for _ in range(50)]
    }
