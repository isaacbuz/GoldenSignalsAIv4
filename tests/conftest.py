"""
Pytest configuration and shared fixtures
"""

import pytest
import asyncio
from typing import Generator
from datetime import datetime, timezone
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=100, freq='D')
    
    # Generate realistic price data
    base_price = 100
    prices = []
    for i in range(100):
        # Add some trend and noise
        trend = i * 0.1
        noise = np.random.normal(0, 2)
        price = base_price + trend + noise
        prices.append(max(price, 10))  # Ensure positive prices
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * np.random.uniform(1.0, 1.02) for p in prices],
        'Low': [p * np.random.uniform(0.98, 1.0) for p in prices],
        'Close': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'Volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # Ensure data integrity
    data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    return data.set_index('Date')


@pytest.fixture
def mock_signal_data():
    """Generate mock signal data"""
    return {
        'id': 'TEST_123',
        'symbol': 'AAPL',
        'action': 'BUY',
        'confidence': 0.75,
        'price': 150.0,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'reason': 'RSI oversold; MACD bullish crossover',
        'indicators': {
            'rsi': 28.5,
            'macd': 1.2,
            'sma_20': 148.3,
            'sma_50': 145.2
        },
        'risk_level': 'medium',
        'entry_price': 150.0,
        'stop_loss': 147.0,
        'take_profit': 156.0,
        'metadata': {
            'volume': 50000000,
            'volatility': 0.02
        },
        'quality_score': 0.85
    }


@pytest.fixture
def mock_market_response():
    """Generate mock market data API response"""
    return {
        'symbol': 'AAPL',
        'name': 'Apple Inc.',
        'price': 150.25,
        'change': 2.50,
        'changePercent': 1.69,
        'dayHigh': 151.00,
        'dayLow': 148.50,
        'volume': 52341234,
        'marketCap': 2500000000000,
        'peRatio': 25.3,
        'week52High': 180.00,
        'week52Low': 120.00,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }


# Test environment configuration
pytest_plugins = []
