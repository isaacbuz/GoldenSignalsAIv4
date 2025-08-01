"""
Unit tests for signal service
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
from datetime import datetime, timezone

from src.services.signal_generation_engine import SignalGenerationEngine
from src.utils.timezone_utils import now_utc

class TestSignalGenerationEngine:
    """Test signal generation engine"""

    @pytest.fixture
    def engine(self):
        """Create engine instance"""
        return SignalGenerationEngine()

    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.cache_ttl == 300
        assert engine.signal_cache == {}
        assert engine.ml_model is not None

    @pytest.mark.asyncio
    async def test_generate_signal_success(self, engine, sample_market_data):
        """Test successful signal generation"""
        with patch.object(engine, '_fetch_market_data', return_value=sample_market_data):
            signal = await engine.generate_signal('AAPL')

            assert signal is not None
            assert signal.symbol == 'AAPL'
            assert signal.action in ['BUY', 'SELL', 'HOLD']
            assert 0 <= signal.confidence <= 1

    @pytest.mark.asyncio
    async def test_generate_signal_with_cache(self, engine, sample_market_data):
        """Test signal generation with caching"""
        with patch.object(engine, '_fetch_market_data', return_value=sample_market_data) as mock_fetch:
            # First call
            signal1 = await engine.generate_signal('AAPL')
            # Second call should use cache
            signal2 = await engine.generate_signal('AAPL')

            # Should only fetch once
            assert mock_fetch.call_count == 1
            assert signal1.id == signal2.id

    def test_analyze_market_data_bullish(self, engine, sample_market_data):
        """Test market analysis for bullish conditions"""
        # Create bullish data
        bullish_data = sample_market_data.copy()
        bullish_data['close'] = bullish_data['close'] * 1.1  # 10% increase

        result = engine._analyze_market_data('AAPL', bullish_data)

        assert result['trend'] > 0
        assert result['strength'] > 0.5

    def test_analyze_market_data_bearish(self, engine, sample_market_data):
        """Test market analysis for bearish conditions"""
        # Create bearish data
        bearish_data = sample_market_data.copy()
        bearish_data['close'] = bearish_data['close'] * 0.9  # 10% decrease

        result = engine._analyze_market_data('AAPL', bearish_data)

        assert result['trend'] < 0

    @pytest.mark.parametrize("symbol,expected", [
        ("", ValueError),
        (None, ValueError),
        ("INVALID SYMBOL!", ValueError),
    ])
    def test_invalid_symbol(self, engine, symbol, expected):
        """Test invalid symbol handling"""
        with pytest.raises(expected):
            engine.generate_signal(symbol)
