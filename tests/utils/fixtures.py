"""Common test fixtures."""

import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_market_data():
    """Create mock market data."""
    from src.ml.models.market_data import MarketData
    return MarketData(
        symbol="TEST",
        current_price=100.0,
        timeframe="1h"
    )

@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis_mock = Mock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.exists.return_value = False
    return redis_mock

@pytest.fixture
def mock_db():
    """Create mock database connection."""
    db_mock = Mock()
    db_mock.execute.return_value = Mock()
    return db_mock
