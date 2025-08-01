"""Comprehensive mock infrastructure for GoldenSignalsAI V2 tests."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.database_url = "sqlite:///test.db"
        self.redis_url = "redis://localhost:6379"
        self.api_key = "test_key"
        self.secret_key = "test_secret"
        self.environment = "test"

class MockDatabaseManager:
    """Mock database manager for testing."""
    def __init__(self):
        self.connection = Mock()
        self.session = Mock()

    def get_session(self):
        return self.session

    def close(self):
        pass

class MockRedisManager:
    """Mock Redis manager for testing."""
    def __init__(self):
        self.client = Mock()
        self.cache = {}

    def get(self, key: str) -> Any:
        return self.cache.get(key)

    def set(self, key: str, value: Any, expire: int = None):
        self.cache[key] = value

    def delete(self, key: str):
        self.cache.pop(key, None)

class MockMarketData:
    """Mock market data for testing."""
    def __init__(self, symbol: str = "AAPL", price: float = 100.0):
        self.symbol = symbol
        self.price = price
        self.bid = price - 0.01
        self.ask = price + 0.01
        self.volume = 1000000
        self.change = 1.0
        self.change_percent = 1.0
        self.timestamp = "2024-01-01T00:00:00Z"

@pytest.fixture
def mock_config():
    """Provide mock configuration."""
    return MockConfig()

@pytest.fixture
def mock_db():
    """Provide mock database manager."""
    return MockDatabaseManager()

@pytest.fixture
def mock_redis():
    """Provide mock Redis manager."""
    return MockRedisManager()

@pytest.fixture
def mock_market_data():
    """Provide mock market data."""
    return MockMarketData()

@pytest.fixture
def mock_agent_dependencies():
    """Provide all mock dependencies for agents."""
    return {
        'config': MockConfig(),
        'db_manager': MockDatabaseManager(),
        'redis_manager': MockRedisManager()
    }
