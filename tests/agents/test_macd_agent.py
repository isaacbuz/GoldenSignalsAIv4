"""
Tests for the MACD (Moving Average Convergence Divergence) agent.
"""
import pytest
import numpy as np
from agents.technical.macd_agent import MACDAgent

def test_macd_initialization():
    """Test MACD agent initialization"""
    agent = MACDAgent(
        name="MACD_Test",
        fast_period=12,
        slow_period=26,
        signal_period=9,
        signal_threshold=0.0
    )
    assert agent.name == "MACD_Test"
    assert agent.agent_type == "technical"
    assert agent.fast_period == 12
    assert agent.slow_period == 26
    assert agent.signal_period == 9
    assert agent.signal_threshold == 0.0

def test_ema_calculation(sample_price_data):
    """Test EMA calculation"""
    agent = MACDAgent()
    ema = agent.calculate_ema(sample_price_data, period=12)

    assert isinstance(ema, np.ndarray)
    assert len(ema) == len(sample_price_data)
    assert not np.isnan(ema).any()

    # First value should equal first price
    assert ema[0] == sample_price_data[0]

    # EMA should smooth the data
    price_std = np.std(sample_price_data)
    ema_std = np.std(ema)
    assert ema_std <= price_std

def test_macd_calculation(sample_price_data):
    """Test MACD calculation"""
    agent = MACDAgent()
    macd_data = agent.calculate_macd(sample_price_data)

    assert "macd_line" in macd_data
    assert "signal_line" in macd_data
    assert "histogram" in macd_data

    macd_line = macd_data["macd_line"]
    signal_line = macd_data["signal_line"]
    histogram = macd_data["histogram"]

    assert len(macd_line) == len(sample_price_data)
    assert len(signal_line) == len(sample_price_data)
    assert len(histogram) == len(sample_price_data)

    # Verify histogram calculation
    np.testing.assert_array_almost_equal(
        histogram,
        macd_line - signal_line
    )

def test_macd_signals(sample_price_data):
    """Test MACD signal generation"""
    agent = MACDAgent(signal_threshold=0.1)

    # Test bullish signal
    rising_prices = [100 + i for i in range(50)]
    result = agent.process({"close_prices": rising_prices})
    assert result["action"] == "buy"
    assert 0 <= result["confidence"] <= 1

    # Test bearish signal
    falling_prices = [100 - i for i in range(50)]
    result = agent.process({"close_prices": falling_prices})
    assert result["action"] == "sell"
    assert 0 <= result["confidence"] <= 1

    # Test neutral signal
    flat_prices = [100] * 50
    result = agent.process({"close_prices": flat_prices})
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0

def test_macd_error_handling():
    """Test MACD agent error handling"""
    agent = MACDAgent()

    # Test missing data
    with pytest.raises(ValueError):
        agent.process({})

    # Test insufficient data
    with pytest.raises(ValueError):
        agent.process({"close_prices": [100] * 10})  # Less than slow_period

    # Test invalid data type
    with pytest.raises(TypeError):
        agent.process({"close_prices": "invalid"})

def test_macd_metadata():
    """Test MACD metadata in results"""
    agent = MACDAgent()
    prices = [100 + i for i in range(50)]

    result = agent.process({"close_prices": prices})
    metadata = result["metadata"]

    assert "macd" in metadata
    assert "signal" in metadata
    assert "histogram" in metadata
    assert "fast_period" in metadata
    assert "slow_period" in metadata
    assert "signal_period" in metadata

    assert metadata["fast_period"] == agent.fast_period
    assert metadata["slow_period"] == agent.slow_period
    assert metadata["signal_period"] == agent.signal_period

def test_macd_threshold_sensitivity():
    """Test MACD signal threshold sensitivity"""
    prices = [100 + i for i in range(50)]

    # Test with zero threshold
    agent_zero = MACDAgent(signal_threshold=0.0)
    result_zero = agent_zero.process({"close_prices": prices})

    # Test with high threshold
    agent_high = MACDAgent(signal_threshold=1.0)
    result_high = agent_high.process({"close_prices": prices})

    # Higher threshold should lead to lower confidence or different action
    assert (result_high["confidence"] <= result_zero["confidence"] or
            result_high["action"] != result_zero["action"])
