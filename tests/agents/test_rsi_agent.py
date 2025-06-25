"""
Tests for the RSI (Relative Strength Index) agent.
"""
import pytest
import numpy as np
from agents.core.technical.momentum.rsi_agent import RSIAgent

def test_rsi_initialization():
    """Test RSI agent initialization"""
    agent = RSIAgent(
        name="RSI_Test",
        period=14,
        overbought=70,
        oversold=30
    )
    assert agent.name == "RSI_Test"
    assert agent.agent_type == "technical"
    assert agent.period == 14
    assert agent.overbought == 70
    assert agent.oversold == 30

def test_rsi_calculation(sample_price_data):
    """Test RSI calculation"""
    agent = RSIAgent()
    rsi = agent.calculate_rsi(sample_price_data)
    
    # RSI should be between 0 and 100
    assert 0 <= rsi <= 100
    
    # Test with insufficient data
    short_data = sample_price_data[:5]
    rsi = agent.calculate_rsi(short_data)
    assert rsi == 50.0  # Default value for insufficient data

def test_rsi_signals(sample_price_data):
    """Test RSI signal generation"""
    agent = RSIAgent(overbought=70, oversold=30)
    
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
    assert result["confidence"] == 0.0

def test_rsi_error_handling():
    """Test RSI agent error handling"""
    agent = RSIAgent()
    
    # Test missing data
    with pytest.raises(ValueError):
        agent.process({})
    
    # Test invalid data type
    with pytest.raises(TypeError):
        agent.process({"close_prices": "invalid"})
    
    # Test empty data
    with pytest.raises(ValueError):
        agent.process({"close_prices": []})

def test_rsi_metadata():
    """Test RSI metadata in results"""
    agent = RSIAgent(period=14, overbought=70, oversold=30)
    prices = [100 + i for i in range(20)]  # Rising prices
    
    result = agent.process({"close_prices": prices})
    metadata = result["metadata"]
    
    assert "rsi" in metadata
    assert "period" in metadata
    assert "overbought" in metadata
    assert "oversold" in metadata
    assert metadata["period"] == 14
    assert metadata["overbought"] == 70
    assert metadata["oversold"] == 30

def test_rsi_confidence_calculation():
    """Test RSI confidence calculation"""
    agent = RSIAgent(overbought=70, oversold=30)
    
    # Test maximum overbought confidence
    extreme_rising = [100 + i*10 for i in range(20)]  # Sharply rising prices
    result = agent.process({"close_prices": extreme_rising})
    assert result["action"] == "sell"
    assert result["confidence"] == 1.0  # Should be capped at 1.0
    
    # Test maximum oversold confidence
    extreme_falling = [100 - i*10 for i in range(20)]  # Sharply falling prices
    result = agent.process({"close_prices": extreme_falling})
    assert result["action"] == "buy"
    assert result["confidence"] == 1.0  # Should be capped at 1.0 