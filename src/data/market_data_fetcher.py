import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries

from config.settings import settings

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetches market data from multiple sources with fallback options"""

    def __init__(self):
        """Initialize the data fetcher"""
        self.alpha_vantage = TimeSeries(key=settings.ALPHA_VANTAGE_KEY, output_format="pandas")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        # Trend Indicators
        df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
        df["MACD"] = ta.trend.macd_diff(df["Close"])

        # Momentum Indicators
        df["RSI"] = ta.momentum.rsi(df["Close"])
        df["Stoch"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])

        # Volatility Indicators
        df["BB_upper"], df["BB_middle"], df["BB_lower"] = ta.volatility.bollinger_bands(df["Close"])
        df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])

        # Volume Indicators
        df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        df["VWAP"] = ta.volume.volume_weighted_average_price(
            df["High"], df["Low"], df["Close"], df["Volume"]
        )

        return df

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data for machine learning"""
        # Price normalization
        df["Close_norm"] = df["Close"] / df["Close"].iloc[0] - 1
        df["Volume_norm"] = df["Volume"] / df["Volume"].rolling(window=20).mean() - 1

        # Technical indicator normalization
        for col in df.columns:
            if col not in ["Date", "Open", "High", "Low", "Close", "Volume", "VWAP"]:
                df[f"{col}_norm"] = (df[col] - df[col].mean()) / df[col].std()

        return df

    def _fetch_yahoo_data(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                logger.error(f"No data found for {symbol} on Yahoo Finance")
                return None

            # Add technical indicators
            df = self._add_technical_indicators(df)

            # Normalize data
            df = self._normalize_data(df)

            return df

        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
            return None

    def _fetch_alpha_vantage_data(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage"""
        try:
            # Convert interval to Alpha Vantage format
            interval_map = {"1d": "daily", "1h": "60min", "5m": "5min"}
            av_interval = interval_map.get(interval, "daily")

            # Fetch data
            df, _ = self.alpha_vantage.get_daily(symbol=symbol, outputsize="full")

            # Filter date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            if df.empty:
                logger.error(f"No data found for {symbol} on Alpha Vantage")
                return None

            # Rename columns to match Yahoo Finance format
            df = df.rename(
                columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",
                    "5. volume": "Volume",
                }
            )

            # Add technical indicators
            df = self._add_technical_indicators(df)

            # Normalize data
            df = self._normalize_data(df)

            return df

        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {str(e)}")
            return None

    def fetch_market_data(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetch market data for a single symbol with fallback options"""
        # Try Yahoo Finance first
        df = self._fetch_yahoo_data(symbol, start_date, end_date, interval)

        # If Yahoo Finance fails, try Alpha Vantage
        if df is None:
            df = self._fetch_alpha_vantage_data(symbol, start_date, end_date, interval)

        if df is None:
            logger.error(f"Failed to fetch data for {symbol} from all sources")

        return df

    def fetch_multiple_symbols(
        self, symbols: List[str], start_date: datetime, end_date: datetime, interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch market data for multiple symbols"""
        data_frames = {}

        for symbol in symbols:
            df = self.fetch_market_data(symbol, start_date, end_date, interval)
            if df is not None:
                data_frames[symbol] = df

        return data_frames
