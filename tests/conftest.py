"""
Pytest configuration and shared fixtures
"""

import pytest
import asyncio
from typing import Generator
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import fixtures to make them available globally
try:
    from tests.fixtures.market_data import *
    from tests.fixtures.agent_mocks import *
except ImportError:
    pass  # Fixtures may not exist yet

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_api: Tests requiring external API access")

# Global test settings
import logging

# Set up logging for tests
logging.basicConfig(level=logging.INFO)

@pytest.fixture(autouse=True)
def reset_test_environment():
    """Reset test environment before each test"""
    yield
    # Cleanup after test if needed
    pass

@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        "test_mode": True,
        "api_key": "test_key",
        "database_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/15",  # Use test database
        "websocket_enabled": False
    }

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


# Create a mock app for testing
def create_test_app():
    """Create a test version of the FastAPI app"""
    app = FastAPI()

    # Add test routes
    @app.get("/api/v1/health")
    async def health_check():
        return {"status": "healthy"}

    @app.get("/api/v1/signals/{symbol}")
    async def get_signals(symbol: str):
        # Mock validation - reject SQL injection attempts
        sql_patterns = ['drop', 'delete', 'union', ';', "'", '"', '--', 'select', 'insert', 'update']
        if any(pattern in symbol.lower() for pattern in sql_patterns):
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="Invalid symbol")
        # Also check for common SQL injection patterns
        if "' or " in symbol.lower() or "'='" in symbol.lower():
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="Invalid symbol")
        return {"signals": [{"symbol": symbol, "confidence": 85, "type": "BUY", "timestamp": "2024-01-19T12:00:00", "source": "test"}]}

    @app.post("/api/v1/analyze")
    async def analyze(data: dict):
        import html
        symbol = data.get("symbol", "")
        # Sanitize XSS attempts more thoroughly
        symbol = html.escape(symbol)
        # Remove dangerous attributes
        dangerous_patterns = ['onerror=', 'onload=', 'onclick=', 'javascript:', 'vbscript:']
        for pattern in dangerous_patterns:
            if pattern in symbol.lower():
                symbol = symbol.replace(pattern, '')
        return {"symbol": symbol, "analysis": "completed"}

    @app.get("/api/v1/portfolio")
    async def get_portfolio():
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Not authenticated")

    @app.get("/api/v1/market-data/{symbol}/current")
    async def get_market_data(symbol: str):
        return {"symbol": symbol, "price": 150.0, "timestamp": "2024-01-19T12:00:00"}

    @app.websocket("/ws/market-data")
    async def websocket_endpoint(websocket):
        await websocket.accept()
        await websocket.send_json({"type": "price", "symbol": "AAPL", "price": 150.0})
        await websocket.close()

    # Add middleware
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        return response

    return app


@pytest.fixture
def test_app():
    """Fixture to provide test app"""
    return create_test_app()
