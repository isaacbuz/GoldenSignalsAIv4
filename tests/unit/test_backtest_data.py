"""
Unit tests for the Backtesting Data Module
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.domain.backtesting.backtest_data import (
    BacktestDataManager,
    MarketDataPoint
)
from src.utils.timezone_utils import now_utc


@pytest.mark.unit
class TestBacktestDataManager:
    """Test cases for the Backtest Data Manager"""

    @pytest.fixture
    def data_manager(self):
        """Create a data manager instance"""
        config = {
            'cache_ttl': 3600,
            'database_url': None,  # Use mock data
            'redis_url': None
        }
        return BacktestDataManager(config)

    @pytest.fixture
    def sample_data_points(self):
        """Create sample market data points"""
        base_time = now_utc()
        points = []

        for i in range(10):
            points.append(MarketDataPoint(
                timestamp=base_time + timedelta(minutes=i*5),
                open=100 + i,
                high=102 + i,
                low=99 + i,
                close=101 + i,
                volume=1000000 + i*10000,
                symbol='AAPL'
            ))

        return points

    def test_market_data_point_to_dict(self, sample_data_points):
        """Test MarketDataPoint to_dict method"""
        point = sample_data_points[0]
        data_dict = point.to_dict()

        assert 'timestamp' in data_dict
        assert data_dict['open'] == point.open
        assert data_dict['high'] == point.high
        assert data_dict['low'] == point.low
        assert data_dict['close'] == point.close
        assert data_dict['volume'] == point.volume
        assert data_dict['symbol'] == point.symbol

    @pytest.mark.asyncio
    async def test_initialization(self, data_manager):
        """Test data manager initialization"""
        await data_manager.initialize()

        # Should initialize without database/redis
        assert data_manager.db_pool is None
        assert data_manager.redis_client is None

    @pytest.mark.asyncio
    async def test_fetch_market_data_mock(self, data_manager):
        """Test fetching market data with mock generation"""
        start_date = now_utc() - timedelta(days=1)
        end_date = now_utc()
        symbols = ['AAPL', 'GOOGL']

        # Fetch data
        market_data = await data_manager.fetch_market_data(
            symbols, start_date, end_date, '5m'
        )

        # Verify results
        assert len(market_data) == 2
        assert 'AAPL' in market_data
        assert 'GOOGL' in market_data

        # Check data structure
        for symbol, df in market_data.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_generate_mock_data(self, data_manager):
        """Test mock data generation"""
        start_date = now_utc() - timedelta(hours=1)
        end_date = now_utc()

        data_points = data_manager._generate_mock_data(
            'AAPL', start_date, end_date, '5m'
        )

        # Should generate ~12 points for 1 hour at 5min intervals
        assert len(data_points) >= 10

        # Check data validity
        for point in data_points:
            assert point.high >= point.low
            assert point.high >= point.open
            assert point.high >= point.close
            assert point.low <= point.open
            assert point.low <= point.close
            assert point.volume > 0

    def test_convert_to_dataframe(self, data_manager, sample_data_points):
        """Test converting data points to DataFrame"""
        df = data_manager._convert_to_dataframe(sample_data_points)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_data_points)
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_convert_empty_data(self, data_manager):
        """Test converting empty data points"""
        df = data_manager._convert_to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_cache_key_generation(self, data_manager):
        """Test cache key generation"""
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 0, 0, 0)

        key = data_manager._get_cache_key('AAPL', start_date, end_date, '5m')

        assert 'backtest:data:AAPL:2024-01-01:2024-01-02:5m' == key

    @pytest.mark.asyncio
    async def test_cache_hit(self, data_manager):
        """Test cache hit scenario"""
        start_date = now_utc() - timedelta(days=1)
        end_date = now_utc()
        symbol = 'AAPL'

        # First fetch - should generate data
        data1 = await data_manager.fetch_market_data(
            [symbol], start_date, end_date, '5m'
        )

        # Second fetch - should hit cache
        data2 = await data_manager.fetch_market_data(
            [symbol], start_date, end_date, '5m'
        )

        # Data should be identical
        pd.testing.assert_frame_equal(data1[symbol], data2[symbol])

    def test_clear_cache(self, data_manager):
        """Test cache clearing"""
        # Add some data to cache
        data_manager._cache['test_key'] = pd.DataFrame({'test': [1, 2, 3]})

        assert len(data_manager._cache) == 1

        # Clear cache
        data_manager.clear_cache()

        assert len(data_manager._cache) == 0

    @pytest.mark.asyncio
    async def test_parallel_fetch(self, data_manager):
        """Test parallel data fetching"""
        start_date = now_utc() - timedelta(days=1)
        end_date = now_utc()
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

        # Time the fetch
        import time
        start_time = time.time()

        market_data = await data_manager.fetch_market_data(
            symbols, start_date, end_date, '5m'
        )

        elapsed = time.time() - start_time

        # Should fetch all symbols
        assert len(market_data) == len(symbols)

        # Parallel fetch should be reasonably fast
        assert elapsed < 2.0  # Should complete in under 2 seconds

    @pytest.mark.asyncio
    async def test_preload_data(self, data_manager):
        """Test data preloading"""
        start_date = now_utc() - timedelta(days=1)
        end_date = now_utc()
        symbols = ['AAPL', 'GOOGL']
        intervals = ['5m', '15m', '1h']

        # Preload data
        await data_manager.preload_data(symbols, start_date, end_date, intervals)

        # Check cache is populated
        expected_entries = len(symbols) * len(intervals)
        assert len(data_manager._cache) >= expected_entries
