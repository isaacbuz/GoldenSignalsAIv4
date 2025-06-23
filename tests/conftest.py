"""
Pytest configuration and fixtures
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock
import asyncio

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your timezone utility
from src.utils.timezone import now_utc

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    dates = pd.date_range(
        start=now_utc() - timedelta(days=30),
        end=now_utc(),
        freq='1h',
        tz='UTC'
    )
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 110, len(dates)),
        'high': np.random.uniform(110, 120, len(dates)),
        'low': np.random.uniform(90, 100, len(dates)),
        'close': np.random.uniform(95, 115, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Ensure high >= open, close, low
    data['high'] = data[['open', 'close', 'low', 'high']].max(axis=1)
    # Ensure low <= open, close, high
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

@pytest.fixture
def sample_signal():
    """Sample trading signal"""
    return {
        'id': 'test_signal_123',
        'symbol': 'AAPL',
        'action': 'BUY',
        'confidence': 0.85,
        'price': 150.25,
        'timestamp': now_utc(),
        'metadata': {
            'indicators': ['RSI', 'MACD'],
            'timeframe': '1h'
        }
    }

@pytest.fixture
def mock_market_service():
    """Mock market data service"""
    service = AsyncMock()
    service.get_market_data.return_value = pd.DataFrame({
        'timestamp': [now_utc()],
        'close': [150.0],
        'volume': [1000000]
    })
    service.get_latest_price.return_value = 150.0
    return service

@pytest.fixture
def mock_signal_repository():
    """Mock signal repository"""
    repo = AsyncMock()
    repo.save_signal.return_value = 'signal_123'
    repo.get_signals.return_value = []
    return repo

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'api': {
            'key': 'test_api_key',
            'secret': 'test_api_secret'
        },
        'cache': {
            'enabled': True,
            'ttl': 60
        },
        'database': {
            'url': 'sqlite:///:memory:'
        },
        'ml': {
            'model_path': 'tests/fixtures/test_model.pkl'
        }
    }

@pytest.fixture
async def async_client():
    """Async HTTP client for testing"""
    from httpx import AsyncClient
    from src.api.app import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Markers
pytest.mark.slow = pytest.mark.slow
pytest.mark.integration = pytest.mark.integration
pytest.mark.unit = pytest.mark.unit
