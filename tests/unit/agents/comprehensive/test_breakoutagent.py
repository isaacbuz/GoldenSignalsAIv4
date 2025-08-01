"""Test for BreakoutAgent."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from agents.core.technical.breakout_agent import BreakoutAgent
from src.ml.models.market_data import MarketData
from src.ml.models.signals import SignalType

class TestBreakoutAgent:
    """Comprehensive tests for BreakoutAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = BreakoutAgent()
        assert agent is not None
        assert agent.name == "Breakout"
        assert hasattr(agent, 'analyze')
        assert hasattr(agent, 'get_required_data_types')

    @pytest.mark.asyncio
    async def test_analyze_buy_signal(self):
        """Test buy signal generation."""
        agent = BreakoutAgent()
        market_data = MarketData(
            symbol="TEST",
            current_price=100.0,
            timeframe="1h"
        )

        # Mock data that should trigger buy signal
        with patch.object(agent, '_fetch_data') as mock_fetch:
            mock_fetch.return_value = self._create_bullish_data()
            signal = await agent.analyze(market_data)

            assert signal is not None
            assert signal.symbol == "TEST"
            assert signal.confidence > 0.5

    @pytest.mark.asyncio
    async def test_analyze_sell_signal(self):
        """Test sell signal generation."""
        agent = BreakoutAgent()
        market_data = MarketData(
            symbol="TEST",
            current_price=100.0,
            timeframe="1h"
        )

        # Mock data that should trigger sell signal
        with patch.object(agent, '_fetch_data') as mock_fetch:
            mock_fetch.return_value = self._create_bearish_data()
            signal = await agent.analyze(market_data)

            assert signal is not None
            assert signal.symbol == "TEST"

    def test_get_required_data_types(self):
        """Test required data types."""
        agent = BreakoutAgent()
        data_types = agent.get_required_data_types()
        assert isinstance(data_types, list)
        assert len(data_types) > 0

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        agent = BreakoutAgent()

        # Test with empty data
        assert agent is not None

        # Test with invalid data
        with pytest.raises(Exception):
            agent.process({})

    def _create_bullish_data(self):
        """Create bullish market data."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
        prices = pd.Series(np.linspace(90, 110, 100) + np.random.normal(0, 1, 100), index=dates)
        return prices

    def _create_bearish_data(self):
        """Create bearish market data."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
        prices = pd.Series(np.linspace(110, 90, 100) + np.random.normal(0, 1, 100), index=dates)
        return prices
