"""
Pytest configuration and shared fixtures for GoldenSignalsAI tests
"""

import pytest
import asyncio
import os
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator
import pandas as pd
import numpy as np

# Test environment setup
os.environ['TESTING'] = 'true'
os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost:5432/test_goldensignals'
os.environ['REDIS_URL'] = 'redis://localhost:6379/1'

# FastAPI testing
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Database
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Local imports
from src.utils.timezone_utils import now_utc


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db_pool():
    """Create a test database connection pool"""
    pool = await asyncpg.create_pool(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        user=os.getenv('DB_USER', 'test'),
        password=os.getenv('DB_PASSWORD', 'test'),
        database=os.getenv('DB_NAME', 'test_goldensignals'),
        min_size=1,
        max_size=5
    )
    yield pool
    await pool.close()


@pytest.fixture
async def db_session():
    """Create a test database session"""
    engine = create_async_engine(
        os.getenv('DATABASE_URL', 'postgresql+asyncpg://test:test@localhost:5432/test_goldensignals'),
        echo=False
    )
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()
        
    await engine.dispose()


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    from simple_backend import app
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client for the FastAPI app"""
    from simple_backend import app
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    
    data = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Ensure high > low and contains open/close
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data


@pytest.fixture
def sample_signal():
    """Create a sample signal for testing"""
    return {
        'id': 'test_signal_001',
        'symbol': 'AAPL',
        'type': 'CALL',
        'strike': 150.0,
        'expiry': (now_utc() + timedelta(days=30)).isoformat(),
        'confidence': 0.85,
        'entry_price': 145.50,
        'target_price': 155.00,
        'stop_loss': 142.00,
        'timeframe': '15m',
        'reasoning': 'Strong momentum with bullish pattern',
        'patterns': ['MACD_BULLISH_CROSS', 'RSI_OVERSOLD_REVERSAL'],
        'urgency': 'HIGH'
    }


@pytest.fixture
def sample_backtest_signal():
    """Create a sample backtest signal"""
    from src.domain.backtesting.advanced_backtest_engine import BacktestSignal
    
    return BacktestSignal(
        timestamp=now_utc(),
        symbol='AAPL',
        action='buy',
        confidence=0.85,
        entry_price=150.00,
        stop_loss=147.00,
        take_profit=156.00,
        position_size=1000,
        agent_scores={'momentum_agent': 0.9, 'sentiment_agent': 0.8},
        agent_reasoning={
            'momentum_agent': 'Strong upward momentum detected',
            'sentiment_agent': 'Positive market sentiment'
        },
        indicators={'RSI': 65, 'MACD': 0.5, 'ATR': 2.5},
        ml_predictions={'price_direction': 0.85, 'volatility': 0.3},
        risk_score=0.2,
        expected_return=0.04,
        sharpe_ratio=1.5
    )


@pytest.fixture
def sample_learning_feedback():
    """Create sample learning feedback"""
    from src.domain.backtesting.adaptive_learning_system import LearningFeedback
    
    return LearningFeedback(
        trade_id='test_trade_001',
        agent_id='momentum_agent',
        signal_timestamp=now_utc(),
        symbol='AAPL',
        action='buy',
        confidence=0.85,
        predicted_return=0.04,
        actual_return=0.03,
        pnl=300.0,
        holding_period=timedelta(hours=2),
        exit_reason='take_profit',
        market_regime='uptrend_normal_vol',
        volatility_level=0.15,
        volume_profile={'relative_volume': 1.2, 'volume_trend': 'increasing'},
        features_at_signal={'momentum': 0.02, 'volume_ratio': 1.2},
        indicators_at_signal={'RSI': 65, 'MACD': 0.5, 'ATR': 2.5},
        accuracy_contribution=1.0,
        sharpe_contribution=1.5,
        reward=0.025,
        regret=0.01,
        surprise=0.01
    )


@pytest.fixture
def mock_redis_client(mocker):
    """Mock Redis client for testing"""
    mock_redis = mocker.AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.exists.return_value = False
    mock_redis.expire.return_value = True
    return mock_redis


@pytest.fixture
def mock_api_responses(mocker):
    """Mock external API responses"""
    # Mock yfinance
    mock_yf = mocker.patch('yfinance.download')
    mock_yf.return_value = pd.DataFrame({
        'Open': [150, 151, 152],
        'High': [152, 153, 154],
        'Low': [149, 150, 151],
        'Close': [151, 152, 153],
        'Volume': [1000000, 1100000, 1200000]
    })
    
    # Mock Alpha Vantage
    mock_av = mocker.patch('alpha_vantage.timeseries.TimeSeries.get_daily')
    mock_av.return_value = ({
        '2024-01-01': {'1. open': '150', '2. high': '152', '3. low': '149', '4. close': '151'},
        '2024-01-02': {'1. open': '151', '2. high': '153', '3. low': '150', '4. close': '152'}
    }, {})
    
    return {
        'yfinance': mock_yf,
        'alpha_vantage': mock_av
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Add any singleton resets here
    yield


@pytest.fixture
def performance_timer():
    """Simple performance timer for tests"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, *args):
            self.elapsed = time.time() - self.start_time
            
    return Timer
