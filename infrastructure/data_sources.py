import abc
import logging

import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries

from config.settings import settings


class BaseMarketDataSource(abc.ABC):
    """
    Abstract base class for market data sources
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{name}")

    @abc.abstractmethod
    async def fetch_data(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """
        Fetch market data for a specific symbol

        Args:
            symbol (str): Stock symbol
            timeframe (str): Data timeframe

        Returns:
            DataFrame with market data
        """

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean DataFrame

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        required_columns = ["open", "high", "low", "close", "volume"]

        # Check for required columns
        if not all(col in df.columns for col in required_columns):
            self.logger.warning(f"Missing required columns in {self.name} data")
            return pd.DataFrame()

        # Convert to numeric and handle potential errors
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with NaN values
        df.dropna(subset=required_columns, inplace=True)

        return df


class YahooFinanceDataSource(BaseMarketDataSource):
    """
    Market data source using Yahoo Finance
    """

    def __init__(self):
        super().__init__("YahooFinance")

    async def fetch_data(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """
        Fetch market data from Yahoo Finance

        Args:
            symbol (str): Stock symbol
            timeframe (str): Data timeframe

        Returns:
            DataFrame with market data
        """
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=timeframe)

            # Add symbol and validate
            data["symbol"] = symbol
            validated_data = self._validate_dataframe(data.reset_index())

            return validated_data

        except Exception as e:
            self.logger.error(f"Yahoo Finance data fetch error for {symbol}: {e}")
            return pd.DataFrame()


class AlphaVantageDataSource(BaseMarketDataSource):
    """
    Market data source using Alpha Vantage
    """

    def __init__(self):
        super().__init__("AlphaVantage")
        self.client = TimeSeries(key=settings.ALPHA_VANTAGE_API_KEY)

    async def fetch_data(self, symbol: str, timeframe: str = "daily") -> pd.DataFrame:
        """
        Fetch market data from Alpha Vantage

        Args:
            symbol (str): Stock symbol
            timeframe (str): Data timeframe

        Returns:
            DataFrame with market data
        """
        try:
            # Fetch daily data
            data, _ = self.client.get_daily(symbol=symbol)

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data, orient="index")
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index)
            df["symbol"] = symbol

            # Validate and return
            return self._validate_dataframe(df.reset_index())

        except Exception as e:
            self.logger.error(f"Alpha Vantage data fetch error for {symbol}: {e}")
            return pd.DataFrame()
