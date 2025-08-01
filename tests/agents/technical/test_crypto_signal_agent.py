"""
Tests for the cryptocurrency signal agent.
"""
import pytest
import pandas as pd
import numpy as np
from agents.technical.crypto.crypto_signal_agent import CryptoSignalAgent

@pytest.fixture
def sample_crypto_data():
    """Generate sample cryptocurrency market data."""
    np.random.seed(42)
    n_points = 100

    # Generate realistic-looking crypto price data with high volatility
    base_price = 100
    returns = np.random.normal(0.001, 0.03, n_points)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate volume with occasional spikes
    base_volume = 1000
    volumes = np.random.lognormal(7, 1, n_points)
    volumes[80:85] *= 5  # Create volume spike

    return {
        "close": prices.tolist(),
        "volume": volumes.tolist(),
        "high": prices * 1.02,
        "low": prices * 0.98,
        "open": prices * 0.99
    }

def test_crypto_agent_initialization():
    """Test CryptoSignalAgent initialization."""
    agent = CryptoSignalAgent(
        name="TestCrypto",
        ema_short=9,
        ema_long=21,
        volume_ma_period=24
    )

    assert agent.name == "TestCrypto"
    assert agent.agent_type == "crypto"
    assert agent.ema_short == 9
    assert agent.ema_long == 21
    assert agent.volume_ma_period == 24

def test_volatility_calculation(sample_crypto_data):
    """Test volatility calculation."""
    agent = CryptoSignalAgent()
    prices = pd.Series(sample_crypto_data["close"])

    volatility = agent.calculate_volatility(prices)
    assert isinstance(volatility, float)
    assert volatility > 0

    # Test empty series
    empty_volatility = agent.calculate_volatility(pd.Series([]))
    assert empty_volatility == 0.0

def test_volume_pump_detection(sample_crypto_data):
    """Test volume pump detection."""
    agent = CryptoSignalAgent()
    volume = pd.Series(sample_crypto_data["volume"])

    # Should detect pump in the spike region
    is_pump = agent.detect_volume_pump(volume)
    assert isinstance(is_pump, bool)

    # Test insufficient data
    assert not agent.detect_volume_pump(pd.Series([100, 200]))

def test_signal_generation(sample_crypto_data):
    """Test signal generation with different market conditions."""
    agent = CryptoSignalAgent()

    # Test normal market conditions
    result = agent.process(sample_crypto_data)
    assert "action" in result
    assert "confidence" in result
    assert "metadata" in result
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1
    assert result["action"] in ["buy", "sell", "hold"]

    # Test high volatility impact
    high_vol_data = sample_crypto_data.copy()
    high_vol_data["close"] = [p * (1 + np.random.normal(0, 0.1)) for p in sample_crypto_data["close"]]
    high_vol_result = agent.process(high_vol_data)
    assert high_vol_result["confidence"] <= result["confidence"]

    # Test volume pump impact
    pump_data = sample_crypto_data.copy()
    pump_data["volume"] = [v * 5 for v in sample_crypto_data["volume"]]
    pump_result = agent.process(pump_data)
    assert pump_result["metadata"]["is_volume_pump"]

def test_error_handling():
    """Test error handling with invalid data."""
    agent = CryptoSignalAgent()

    # Test missing data
    result = agent.process({})
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0
    assert "error" in result["metadata"]

    # Test invalid data types
    invalid_data = {
        "close": "invalid",
        "volume": None
    }
    result = agent.process(invalid_data)
    assert result["action"] == "hold"
    assert result["confidence"] == 0.0
    assert "error" in result["metadata"]

def test_edge_cases(sample_crypto_data):
    """Test edge cases and boundary conditions."""
    agent = CryptoSignalAgent()

    # Test single data point
    single_point = {k: [v[0]] for k, v in sample_crypto_data.items()}
    result = agent.process(single_point)
    assert result["action"] == "hold"

    # Test flat prices
    flat_data = sample_crypto_data.copy()
    flat_data["close"] = [100] * len(sample_crypto_data["close"])
    result = agent.process(flat_data)
    assert result["action"] == "hold"

    # Test extreme values
    extreme_data = sample_crypto_data.copy()
    extreme_data["close"][-1] = extreme_data["close"][-2] * 2  # 100% price jump
    result = agent.process(extreme_data)
    assert result["metadata"]["volatility"] > agent.volatility_threshold
