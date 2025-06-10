import os
import sys
import pytest
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.orchestration.orchestrator import AgentOrchestrator
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.sentiment.sentiment_agent import SentimentAgent
from agents.backtesting.backtest_engine import BacktestEngine

@pytest.fixture(scope='session')
def sample_market_data():
    """Generate synthetic market data for testing."""
    np.random.seed(42)
    return {
        'stock_prices': np.random.normal(100, 20, (100, 5)),
        'options_volume': np.random.poisson(500, (100, 3)),
        'sentiment_scores': np.random.uniform(-1, 1, 100)
    }

@pytest.fixture(scope='session')
def ml_model_fixtures():
    """Provide fixtures for machine learning model testing."""
    from src.domain.models.ai_models import LSTMModel, TransformerModel
    
    lstm_model = LSTMModel(input_dim=5, hidden_dim=32)
    transformer_model = TransformerModel(input_dim=5, hidden_dim=32)
    
    return {
        'lstm': lstm_model,
        'transformer': transformer_model
    }

@pytest.fixture(scope='session')
def torch_device():
    """Determine and return the appropriate torch device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", 
        "performance: mark test related to performance metrics"
    )

@pytest.fixture
def sample_price_data() -> List[float]:
    """Generate sample price data for testing"""
    np.random.seed(42)  # For reproducibility
    prices = [100.0]  # Start price
    for _ in range(99):  # Generate 100 points total
        change = np.random.normal(0, 1)  # Random price change
        prices.append(max(0.1, prices[-1] * (1 + change/100)))  # Ensure price > 0
    return prices

@pytest.fixture
def sample_news_data() -> List[str]:
    """Generate sample news data for testing"""
    return [
        "Company reports strong earnings, beating expectations",
        "Market uncertainty leads to volatility",
        "New product launch receives positive reviews",
        "Industry faces regulatory challenges",
        "Company announces strategic partnership"
    ]

@pytest.fixture
def rsi_agent() -> RSIAgent:
    """Create RSI agent for testing"""
    return RSIAgent(
        name="RSI_Test",
        period=14,
        overbought=70,
        oversold=30
    )

@pytest.fixture
def macd_agent() -> MACDAgent:
    """Create MACD agent for testing"""
    return MACDAgent(
        name="MACD_Test",
        fast_period=12,
        slow_period=26,
        signal_period=9
    )

@pytest.fixture
def sentiment_agent() -> SentimentAgent:
    """Create sentiment agent for testing"""
    return SentimentAgent(name="Sentiment_Test")

@pytest.fixture
def orchestrator(rsi_agent, macd_agent, sentiment_agent) -> AgentOrchestrator:
    """Create orchestrator with all agents for testing"""
    orchestrator = AgentOrchestrator()
    for agent in [rsi_agent, macd_agent, sentiment_agent]:
        orchestrator.register_agent(agent)
    return orchestrator

@pytest.fixture
def backtest_engine(orchestrator) -> BacktestEngine:
    """Create backtest engine for testing"""
    return BacktestEngine(
        orchestrator=orchestrator,
        initial_capital=100000.0,
        commission=0.001
    )

@pytest.fixture
def market_data(sample_price_data, sample_news_data) -> Dict[str, Any]:
    """Create sample market data for testing"""
    return {
        "close_prices": sample_price_data,
        "texts": sample_news_data,
        "timestamp": datetime.now().isoformat()
    }

@pytest.fixture
def historical_data() -> pd.DataFrame:
    """Create historical market data for backtesting"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    prices = []
    price = 100.0
    for _ in range(len(dates)):
        change = np.random.normal(0, 1)
        price *= (1 + change/100)
        prices.append(max(0.1, price))
    
    return pd.DataFrame({
        'Close': prices,
        'Open': [p * (1 - np.random.uniform(-0.01, 0.01)) for p in prices],
        'High': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

@pytest.fixture
def signal_weights() -> Dict[str, float]:
    """Default signal weights for testing"""
    return {
        "technical": 0.6,
        "sentiment": 0.4
    }

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing technical indicators."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    return pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.normal(1000000, 200000, len(dates))
    }).set_index('date')

@pytest.fixture
def sample_news_data():
    """Generate sample news data for sentiment analysis testing."""
    return [
        {
            'timestamp': '2023-01-01T00:00:00Z',
            'headline': 'Company XYZ Reports Strong Q4 Earnings',
            'content': 'Company XYZ exceeded analyst expectations...',
            'source': 'Financial Times'
        },
        {
            'timestamp': '2023-01-02T00:00:00Z',
            'headline': 'Market Volatility Increases Amid Global Concerns',
            'content': 'Global markets experienced increased volatility...',
            'source': 'Reuters'
        }
    ]

@pytest.fixture
def sample_social_data():
    """Generate sample social media data for sentiment analysis testing."""
    return [
        {
            'timestamp': '2023-01-01T00:00:00Z',
            'platform': 'Twitter',
            'text': 'Bullish on $XYZ after strong earnings report! #stocks',
            'user': 'trader123',
            'followers': 1000
        },
        {
            'timestamp': '2023-01-02T00:00:00Z',
            'platform': 'StockTwits',
            'text': 'Market looking bearish today... $SPY',
            'user': 'investor456',
            'followers': 2000
        }
    ]

@pytest.fixture
def test_model_registry(tmp_path):
    """Create a temporary model registry for testing."""
    registry_path = tmp_path / "models"
    registry_path.mkdir()
    return str(registry_path)

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing trading strategies."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate price data with a trend and some noise
    trend = np.linspace(100, 120, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    prices = trend + noise
    
    return pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.normal(1000000, 200000, len(dates)),
        'sentiment': np.random.normal(0, 1, len(dates))
    }).set_index('date')

@pytest.fixture(scope="session")
def market_calendar():
    """Create a market calendar for testing."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    return dates

@pytest.fixture(scope="session")
def risk_free_rates():
    """Create risk-free rates for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    rates = pd.Series(
        np.random.normal(0.02/252, 0.001, len(dates)),  # Daily rates
        index=dates
    )
    return rates

@pytest.fixture(scope="session")
def market_factors():
    """Create market factor returns for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    factors = pd.DataFrame({
        'market': np.random.normal(0.0001, 0.01, len(dates)),
        'size': np.random.normal(0, 0.005, len(dates)),
        'value': np.random.normal(0, 0.005, len(dates)),
        'momentum': np.random.normal(0, 0.006, len(dates)),
        'volatility': np.random.normal(0, 0.004, len(dates))
    }, index=dates)
    return factors

@pytest.fixture(scope="session")
def sample_portfolio():
    """Create a sample portfolio configuration for testing."""
    return {
        'assets': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        'weights': [0.25, 0.25, 0.25, 0.25],
        'risk_limits': {
            'max_position': 0.3,
            'max_sector_exposure': 0.4,
            'var_limit': 0.02,
            'tracking_error_limit': 0.03
        },
        'rebalancing': {
            'frequency': 'monthly',
            'threshold': 0.05
        }
    }

@pytest.fixture(scope="session")
def technical_indicators():
    """Create technical indicator configurations for testing."""
    return {
        'moving_averages': [5, 10, 20, 50, 200],
        'rsi': {'period': 14},
        'macd': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        },
        'bollinger_bands': {
            'period': 20,
            'std_dev': 2
        }
    }

@pytest.fixture(scope="session")
def model_configs():
    """Create model configurations for testing."""
    return {
        'time_series': {
            'lookback': 10,
            'horizon': 5,
            'batch_size': 32,
            'epochs': 100
        },
        'factor': {
            'estimation_window': 252,
            'min_periods': 126
        },
        'ensemble': {
            'n_models': 5,
            'aggregation': 'weighted_average'
        }
    }

@pytest.fixture(scope="session")
def market_scenarios():
    """Create market scenario data for stress testing."""
    base_volatility = 0.15
    scenarios = {
        'normal': {
            'returns': np.random.normal(0.0001, base_volatility/np.sqrt(252), 252),
            'volatility': base_volatility
        },
        'high_volatility': {
            'returns': np.random.normal(0.0001, base_volatility*2/np.sqrt(252), 252),
            'volatility': base_volatility * 2
        },
        'market_crash': {
            'returns': np.random.normal(-0.002, base_volatility*3/np.sqrt(252), 252),
            'volatility': base_volatility * 3
        },
        'bull_market': {
            'returns': np.random.normal(0.001, base_volatility/np.sqrt(252), 252),
            'volatility': base_volatility * 0.8
        }
    }
    return scenarios
