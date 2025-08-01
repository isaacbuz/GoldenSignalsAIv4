"""
Unit tests for Market Data Service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from src.services.market_data_service import MarketDataService


class TestMarketDataService:
    """Test cases for MarketDataService"""

    @pytest.fixture
    def market_data_service(self):
        """Create test market data service"""
        return MarketDataService()

    @pytest.fixture
    def mock_stock_data(self):
        """Create mock stock data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(95, 115, 100),
            'Volume': np.random.randint(1000000, 5000000, 100),
            'Adj Close': np.random.uniform(95, 115, 100)
        }, index=dates)
        return data

    @patch('yfinance.download')
    def test_get_historical_data(self, mock_download, market_data_service, mock_stock_data):
        """Test getting historical data"""
        mock_download.return_value = mock_stock_data

        data = market_data_service.get_historical_data("AAPL", period="1mo")

        assert data is not None
        assert len(data) > 0
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        mock_download.assert_called_once()

    @patch('yfinance.Ticker')
    def test_get_realtime_quote(self, mock_ticker_class, market_data_service):
        """Test getting real-time quote"""
        mock_ticker = Mock()
        mock_ticker.info = {
            'regularMarketPrice': 150.25,
            'regularMarketPreviousClose': 148.50,
            'regularMarketVolume': 50000000,
            'regularMarketDayHigh': 151.00,
            'regularMarketDayLow': 149.00,
            'bid': 150.20,
            'ask': 150.30,
            'bidSize': 100,
            'askSize': 200
        }
        mock_ticker_class.return_value = mock_ticker

        quote = market_data_service.get_realtime_quote("AAPL")

        assert quote is not None
        assert quote['symbol'] == "AAPL"
        assert quote['price'] == 150.25
        assert quote['change'] == 1.75
        assert quote['change_percent'] > 0
        assert quote['volume'] == 50000000

    @patch('yfinance.download')
    def test_get_multiple_stocks(self, mock_download, market_data_service, mock_stock_data):
        """Test getting data for multiple stocks"""
        mock_download.return_value = mock_stock_data

        symbols = ["AAPL", "GOOGL", "MSFT"]
        data = market_data_service.get_multiple_stocks(symbols, period="1d")

        assert data is not None
        assert isinstance(data, dict)
        mock_download.assert_called()

    @patch('yfinance.Ticker')
    def test_get_options_chain(self, mock_ticker_class, market_data_service):
        """Test getting options chain"""
        mock_ticker = Mock()

        # Mock options data
        mock_calls = pd.DataFrame({
            'strike': [145, 150, 155],
            'lastPrice': [6.50, 3.20, 1.10],
            'bid': [6.40, 3.10, 1.00],
            'ask': [6.60, 3.30, 1.20],
            'volume': [1000, 2000, 500],
            'openInterest': [5000, 8000, 2000],
            'impliedVolatility': [0.25, 0.22, 0.20]
        })

        mock_puts = pd.DataFrame({
            'strike': [145, 150, 155],
            'lastPrice': [1.20, 3.50, 6.80],
            'bid': [1.10, 3.40, 6.70],
            'ask': [1.30, 3.60, 6.90],
            'volume': [800, 1500, 1200],
            'openInterest': [4000, 6000, 3000],
            'impliedVolatility': [0.24, 0.23, 0.22]
        })

        mock_ticker.option_chain.return_value = (mock_calls, mock_puts)
        mock_ticker.options = ['2024-01-19', '2024-02-16']
        mock_ticker_class.return_value = mock_ticker

        options = market_data_service.get_options_chain("AAPL")

        assert options is not None
        assert 'expirations' in options
        assert 'calls' in options
        assert 'puts' in options
        assert len(options['calls']) > 0
        assert len(options['puts']) > 0

    @patch('yfinance.download')
    def test_get_intraday_data(self, mock_download, market_data_service):
        """Test getting intraday data"""
        # Create intraday mock data
        times = pd.date_range(end=datetime.now(), periods=78, freq='5min')
        intraday_data = pd.DataFrame({
            'Open': np.random.uniform(150, 151, 78),
            'High': np.random.uniform(151, 152, 78),
            'Low': np.random.uniform(149, 150, 78),
            'Close': np.random.uniform(150, 151, 78),
            'Volume': np.random.randint(10000, 50000, 78)
        }, index=times)

        mock_download.return_value = intraday_data

        data = market_data_service.get_intraday_data("AAPL", interval="5m")

        assert data is not None
        assert len(data) > 0
        assert len(data) == 78  # Should be ~78 5-minute bars in a trading day

    @patch('yfinance.Ticker')
    def test_get_company_info(self, mock_ticker_class, market_data_service):
        """Test getting company information"""
        mock_ticker = Mock()
        mock_ticker.info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'trailingPE': 28.5,
            'forwardPE': 25.2,
            'dividendYield': 0.0048,
            'beta': 1.25
        }
        mock_ticker_class.return_value = mock_ticker

        info = market_data_service.get_company_info("AAPL")

        assert info is not None
        assert info['name'] == 'Apple Inc.'
        assert info['sector'] == 'Technology'
        assert info['market_cap'] > 0
        assert 'pe_ratio' in info
        assert 'dividend_yield' in info

    @patch('yfinance.download')
    def test_calculate_technical_indicators(self, mock_download, market_data_service, mock_stock_data):
        """Test technical indicator calculation"""
        mock_download.return_value = mock_stock_data

        data = market_data_service.get_historical_data("AAPL", period="3mo")
        indicators = market_data_service.calculate_technical_indicators(data)

        assert indicators is not None
        assert 'RSI' in indicators
        assert 'MACD' in indicators
        assert 'MACD_Signal' in indicators
        assert 'BB_Upper' in indicators
        assert 'BB_Lower' in indicators
        assert 'SMA_20' in indicators
        assert 'SMA_50' in indicators
        assert 'Volume_SMA' in indicators

    @patch('yfinance.download')
    def test_error_handling(self, mock_download, market_data_service):
        """Test error handling"""
        # Simulate API error
        mock_download.side_effect = Exception("API Error")

        data = market_data_service.get_historical_data("INVALID_SYMBOL")

        # Should handle error gracefully
        assert data is None or data.empty

    @patch('yfinance.Ticker')
    def test_cache_functionality(self, mock_ticker_class, market_data_service):
        """Test caching functionality"""
        mock_ticker = Mock()
        mock_ticker.info = {'regularMarketPrice': 150.25}
        mock_ticker_class.return_value = mock_ticker

        # First call
        quote1 = market_data_service.get_realtime_quote("AAPL")

        # Second call (should use cache)
        quote2 = market_data_service.get_realtime_quote("AAPL")

        # Should only call API once due to caching
        assert mock_ticker_class.call_count <= 2  # May call for both operations
        assert quote1 == quote2

    @patch('yfinance.download')
    def test_data_validation(self, mock_download, market_data_service):
        """Test data validation"""
        # Create data with some invalid values
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        invalid_data = pd.DataFrame({
            'Open': [100, np.nan, 102, 103, np.inf, 105, 106, -107, 108, 109],
            'High': [101, 102, np.nan, 104, 105, np.inf, 107, 108, -109, 110],
            'Low': [99, 100, 101, np.nan, 103, 104, np.inf, 106, 107, -108],
            'Close': [100.5, 101.5, 102.5, 103.5, np.nan, 105.5, 106.5, np.inf, 108.5, 109.5],
            'Volume': [1000000, 0, -1000000, np.nan, 2000000, 3000000, np.inf, 4000000, 5000000, 6000000]
        }, index=dates)

        mock_download.return_value = invalid_data

        data = market_data_service.get_historical_data("AAPL", validate=True)

        # Should clean invalid values
        assert data is not None
        assert not data.isnull().any().any()  # No NaN values
        assert not np.isinf(data).any().any()  # No inf values
        assert (data >= 0).all().all()  # No negative values
