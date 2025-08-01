"""
Tests for the RSI (Relative Strength Index) agent.
"""
import pytest
import numpy as np
import pandas as pd
from agents.core.technical.momentum.rsi_agent import RSIAgent
from agents.base import AgentConfig
from src.core.database import DatabaseManager
from src.core.redis_manager import RedisManager
from src.ml.models.market_data import MarketData

@pytest.fixture
def agent_setup():
    """Create test agent with dependencies"""
    config = AgentConfig(name="RSI_Test", version="1.0.0")
    db_manager = DatabaseManager()
    redis_manager = RedisManager()
    return config, db_manager, redis_manager

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    # Generate 50 random prices starting at 100
    np.random.seed(42)
    prices = [100]
    for _ in range(49):
        change = np.random.normal(0, 1)
        prices.append(prices[-1] * (1 + change/100))
    return prices

def test_rsi_initialization(agent_setup):
    """Test RSI agent initialization"""
    config, db_manager, redis_manager = agent_setup

    agent = RSIAgent(
        config=config,
        db_manager=db_manager,
        redis_manager=redis_manager,
        period=14,
        overbought=70,
        oversold=30
    )
    assert agent.config.name == "RSI_Test"
    assert agent.period == 14
    assert agent.overbought == 70
    assert agent.oversold == 30

def test_rsi_calculation(agent_setup, sample_price_data):
    """Test RSI calculation"""
    config, db_manager, redis_manager = agent_setup
    agent = RSIAgent(config=config, db_manager=db_manager, redis_manager=redis_manager)

    prices_series = pd.Series(sample_price_data)
    rsi = agent.calculate_rsi(prices_series)

    # RSI should be between 0 and 100
    assert rsi is not None
    assert 0 <= rsi <= 100

    # Test with insufficient data
    short_data = pd.Series(sample_price_data[:5])
    rsi = agent.calculate_rsi(short_data)
    assert rsi is None  # Should return None for insufficient data

def test_rsi_signals(agent_setup, sample_price_data):
    """Test RSI signal generation"""
    config, db_manager, redis_manager = agent_setup
    agent = RSIAgent(
        config=config,
        db_manager=db_manager,
        redis_manager=redis_manager,
        overbought=70,
        oversold=30
    )

    # Test overbought condition
    rising_prices = [100 + i for i in range(20)]  # Steadily rising prices
    result = agent.process({"close_prices": rising_prices})
    assert result["action"] == "sell"
    assert 0 <= result["confidence"] <= 1

    # Test oversold condition
    falling_prices = [100 - i for i in range(20)]  # Steadily falling prices
    result = agent.process({"close_prices": falling_prices})
    assert result["action"] == "buy"
    assert 0 <= result["confidence"] <= 1

    # Test neutral condition
    flat_prices = [100] * 20  # Flat prices
    result = agent.process({"close_prices": flat_prices})
    assert result["action"] == "hold"
    assert result["confidence"] >= 0.0

def test_rsi_error_handling(agent_setup):
    """Test RSI agent error handling"""
    config, db_manager, redis_manager = agent_setup
    agent = RSIAgent(config=config, db_manager=db_manager, redis_manager=redis_manager)

    # Test missing data
    result = agent.process({})
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0
    assert "error" in result["metadata"]

    # Test invalid data type
    result = agent.process({"close_prices": "invalid"})
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0

    # Test empty data
    result = agent.process({"close_prices": []})
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0

def test_rsi_metadata(agent_setup):
    """Test RSI metadata in results"""
    config, db_manager, redis_manager = agent_setup
    agent = RSIAgent(
        config=config,
        db_manager=db_manager,
        redis_manager=redis_manager,
        period=14,
        overbought=70,
        oversold=30
    )
    prices = [100 + i for i in range(20)]  # Rising prices

    result = agent.process({"close_prices": prices})
    metadata = result["metadata"]

    assert "rsi" in metadata
    assert "period" in metadata
    assert "overbought" in metadata
    assert "oversold" in metadata
    assert metadata["period"] == 14
    # Note: overbought/oversold may be adjusted by trend_factor
    assert "overbought" in metadata
    assert "oversold" in metadata

def test_rsi_confidence_calculation(agent_setup):
    """Test RSI confidence calculation"""
    config, db_manager, redis_manager = agent_setup
    agent = RSIAgent(
        config=config,
        db_manager=db_manager,
        redis_manager=redis_manager,
        overbought=70,
        oversold=30
    )

    # Test maximum overbought confidence
    extreme_rising = [100 + i*10 for i in range(20)]  # Sharply rising prices
    result = agent.process({"close_prices": extreme_rising})
    assert result["action"] == "sell"
    assert result["confidence"] <= 1.0  # Should be capped at 1.0

    # Test maximum oversold confidence
    extreme_falling = [100 - i*5 for i in range(20)]  # Sharply falling prices
    result = agent.process({"close_prices": extreme_falling})
    assert result["action"] == "buy"
    assert result["confidence"] <= 1.0  # Should be capped at 1.0

@pytest.mark.asyncio
async def test_rsi_analyze(agent_setup, sample_price_data):
    """Test RSI analyze method"""
    config, db_manager, redis_manager = agent_setup
    agent = RSIAgent(config=config, db_manager=db_manager, redis_manager=redis_manager)

    # Create market data with close prices
    market_data = MarketData(
        symbol="AAPL",
        data={"close_prices": sample_price_data}
    )
    market_data.current_price = sample_price_data[-1]

    signal = await agent.analyze(market_data)

    assert signal.symbol == "AAPL"
    assert signal.signal_type.value in ["buy", "sell", "hold"]
    assert 0 <= signal.confidence <= 1
    assert signal.source.value == "technical_analysis"
    assert "rsi" in signal.features

def test_get_required_data_types(agent_setup):
    """Test that agent returns correct required data types"""
    config, db_manager, redis_manager = agent_setup
    agent = RSIAgent(config=config, db_manager=db_manager, redis_manager=redis_manager)

    data_types = agent.get_required_data_types()
    assert isinstance(data_types, list)
    assert "price" in data_types
    assert "close_prices" in data_types
