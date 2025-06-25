"""Unit tests for RSI Agent"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from agents.core.technical.momentum.rsi_agent import RSIAgent

class TestRSIAgent:
    """Test suite for RSI Agent"""
    
    @pytest.fixture
    def rsi_agent(self):
        """Create RSI agent instance"""
        return RSIAgent(name="TestRSI", period=14)
    
    @pytest.fixture
    def simple_price_data(self):
        """Simple price data for testing"""
        return {
            "close_prices": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                            111, 110, 112, 114, 113, 115, 117, 116, 118, 120]
        }
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = RSIAgent(name="TestRSI", period=14, oversold=30, overbought=70)
        
        assert agent.name == "TestRSI"
        assert agent.agent_type == "technical"
        assert agent.period == 14
        assert agent.oversold_threshold == 30
        assert agent.overbought_threshold == 70
    
    def test_rsi_calculation(self, rsi_agent):
        """Test RSI calculation logic"""
        # Create price series with known pattern
        prices = pd.Series([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10,
                           45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
                           46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
        
        rsi = rsi_agent.calculate_rsi(prices)
        
        # RSI should be a float
        assert isinstance(rsi, float)
        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100
    
    def test_oversold_signal(self, rsi_agent):
        """Test oversold buy signal generation"""
        # Create price data that will result in oversold RSI
        prices = []
        base = 100
        # Generate declining prices
        for i in range(20):
            base -= 2
            prices.append(base)
        
        result = rsi_agent.process({"close_prices": prices})
        
        assert result["action"] == "buy"
        assert result["confidence"] > 0
        assert "rsi" in result["metadata"]
        assert result["metadata"]["signal"] == "oversold"
    
    def test_overbought_signal(self, rsi_agent):
        """Test overbought sell signal generation"""
        # Create price data that will result in overbought RSI
        prices = []
        base = 100
        # Generate rising prices
        for i in range(20):
            base += 2
            prices.append(base)
        
        result = rsi_agent.process({"close_prices": prices})
        
        assert result["action"] == "sell"
        assert result["confidence"] > 0
        assert "rsi" in result["metadata"]
        assert result["metadata"]["signal"] == "overbought"
    
    def test_neutral_signal(self, rsi_agent, simple_price_data):
        """Test neutral/hold signal"""
        result = rsi_agent.process(simple_price_data)
        
        assert result["action"] == "hold"
        assert result["confidence"] == 0.0
        assert "rsi" in result["metadata"]
        assert result["metadata"]["signal"] == "neutral"
    
    def test_insufficient_data(self, rsi_agent):
        """Test handling of insufficient data"""
        # Less than period + 1 data points
        result = rsi_agent.process({"close_prices": [100, 101, 102]})
        
        assert result["action"] == "hold"
        assert result["confidence"] == 0.0
        assert "error" in result["metadata"]
    
    def test_invalid_data_handling(self, rsi_agent):
        """Test handling of invalid data"""
        # Missing required field
        result = rsi_agent.process({"open": [100, 101, 102]})
        
        assert result["action"] == "hold"
        assert result["confidence"] == 0.0
        assert "error" in result["metadata"]
    
    def test_confidence_calculation(self, rsi_agent):
        """Test confidence score calculation"""
        # Create data with RSI = 20 (oversold)
        prices = [100, 98, 96, 94, 92, 90, 88, 86, 84, 82,
                 80, 78, 76, 74, 72, 70, 68, 66, 64, 62]
        
        result = rsi_agent.process({"close_prices": prices})
        
        # With RSI = 20, confidence should be (30-20)/30 = 0.33
        assert result["action"] == "buy"
        assert 0.3 <= result["confidence"] <= 0.4
    
    @pytest.mark.parametrize("period,expected", [
        (7, "technical"),
        (14, "technical"),
        (21, "technical"),
        (28, "technical")
    ])
    def test_different_periods(self, period, expected, simple_price_data):
        """Test agent with different RSI periods"""
        agent = RSIAgent(period=period)
        result = agent.process(simple_price_data)
        
        assert agent.agent_type == expected
        assert "rsi" in result["metadata"]
        assert result["metadata"]["period"] == period
    
    def test_edge_cases(self, rsi_agent):
        """Test edge cases"""
        # All same prices
        same_prices = {"close_prices": [100] * 20}
        result = rsi_agent.process(same_prices)
        assert result["action"] == "hold"
        
        # Empty data
        empty_data = {"close_prices": []}
        result = rsi_agent.process(empty_data)
        assert result["action"] == "hold"
        assert "error" in result["metadata"]
    
    @patch('logging.Logger.error')
    def test_error_logging(self, mock_logger, rsi_agent):
        """Test that errors are properly logged"""
        # Cause an error with invalid data type
        rsi_agent.process({"close_prices": "invalid"})
        
        # Verify error was logged
        mock_logger.assert_called_once() 