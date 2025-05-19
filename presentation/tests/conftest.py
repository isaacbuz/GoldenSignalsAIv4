import pytest
import pandas as pd
from unittest.mock import patch
from domain.trading.strategies.trading_env import TradingEnv
from infrastructure.data.fetchers.database_fetcher import fetch_stock_data

@pytest.fixture
def symbol():
    return "AAPL"

@pytest.fixture(autouse=True)
def mock_fetch_stock_data():
    with patch('infrastructure.data.fetchers.database_fetcher.fetch_stock_data') as mocked_fetch:
        mock_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'open': [100 + i for i in range(100)],
            'high': [105 + i for i in range(100)],
            'low': [95 + i for i in range(100)],
            'close': [100 + i for i in range(100)],
            'volume': [1000000] * 100
        })
        mocked_fetch.return_value = mock_data
        yield mocked_fetch

@pytest.fixture
def stock_data(symbol, mock_fetch_stock_data):
    df = fetch_stock_data(symbol)
    if df is None:
        pytest.fail(f"Failed to fetch stock data for {symbol}")
    return df

@pytest.fixture
def trading_env(stock_data, symbol):
    try:
        return TradingEnv(stock_data, symbol)
    except Exception as e:
        pytest.fail(f"Failed to initialize TradingEnv: {str(e)}")
