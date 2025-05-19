# infrastructure/data_fetcher.py
# Purpose: Fetches market data (stock prices, options data, news, and social media sentiment)
# from external APIs for use in GoldenSignalsAI. Includes mock implementations for testing.

import logging
import os
from typing import Dict, List

import pandas as pd
import requests
import yfinance as yf

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetches market data from external APIs."""

    def __init__(self, api_key=None):
        # Accept api_key for DI compatibility, but fallback to env var if not provided
        self.alpha_vantage_api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        logger.info({"message": "DataFetcher initialized"})

    def fetch_stock_data(
        self, symbol: str, interval: str = "1d", period: str = "1mo"
    ) -> pd.DataFrame:
        """Fetch historical stock data using yfinance.

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL').
            interval (str): Data interval ('1m', '1d', etc.).
            period (str): Data period ('1d', '1mo', etc.).

        Returns:
            pd.DataFrame: Stock data with columns like 'Open', 'High', 'Low', 'Close', 'Volume'.
        """
        logger.info({"message": f"Fetching stock data for {symbol}"})
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                logger.warning({"message": f"No stock data found for {symbol}"})
                return pd.DataFrame()
            logger.info(
                {"message": f"Fetched {len(data)} rows of stock data for {symbol}"}
            )
            return data
        except Exception as e:
            logger.error(
                {"message": f"Failed to fetch stock data for {symbol}: {str(e)}"}
            )
            # Mock data for testing
            return pd.DataFrame(
                {
                    "Open": [100],
                    "High": [102],
                    "Low": [99],
                    "Close": [101],
                    "Volume": [1000000],
                },
                index=pd.date_range(start="2025-05-01", periods=1),
            )

    def fetch_options_data(self, symbol: str) -> pd.DataFrame:
        """Fetch options chain data (mock implementation; replace with actual API in production).

        Args:
            symbol (str): Stock symbol.

        Returns:
            pd.DataFrame: Options data with columns like 'strike', 'iv', 'delta'.
        """
        logger.info({"message": f"Fetching options data for {symbol}"})
        try:
            # Mock options data (in practice, use an API like Alpha Vantage or a broker API)
            data = pd.DataFrame(
                {
                    "timestamp": ["2025-05-01"],
                    "symbol": [symbol],
                    "call_volume": [1000],
                    "put_volume": [500],
                    "call_oi": [2000],
                    "put_oi": [1500],
                    "strike": [100],
                    "iv": [0.3],
                    "delta": [0.5],
                    "gamma": [0.1],
                    "theta": [-0.02],
                    "quantity": [10],
                    "call_put": ["call"],
                }
            )
            logger.info({"message": f"Fetched mock options data for {symbol}"})
            return data
        except Exception as e:
            logger.error(
                {"message": f"Failed to fetch options data for {symbol}: {str(e)}"}
            )
            return pd.DataFrame()

    def fetch_news_data(self, symbol: str) -> List[Dict]:
        """Fetch news articles related to the stock symbol.

        Args:
            symbol (str): Stock symbol.

        Returns:
            List[Dict]: List of news articles.
        """
        logger.info({"message": f"Fetching news data for {symbol}"})
        try:
            if not self.news_api_key:
                raise ValueError("NEWS_API_KEY not set")
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.news_api_key}"
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            logger.info(
                {"message": f"Fetched {len(articles)} news articles for {symbol}"}
            )
            return articles
        except Exception as e:
            logger.error(
                {"message": f"Failed to fetch news data for {symbol}: {str(e)}"}
            )
            return [{"title": "Mock News", "description": "This is a mock article."}]

    def fetch_social_sentiment(self, symbol: str) -> List[Dict]:
        """Fetch social media sentiment (mock implementation; replace with Twitter API in production).

        Args:
            symbol (str): Stock symbol.

        Returns:
            List[Dict]: List of social media posts with sentiment scores.
        """
        logger.info({"message": f"Fetching social sentiment for {symbol}"})
        try:
            # Mock social sentiment data
            data = [{"text": f"Positive sentiment for {symbol}", "sentiment": 0.7}]
            logger.info({"message": f"Fetched mock social sentiment for {symbol}"})
            return data
        except Exception as e:
            logger.error(
                {"message": f"Failed to fetch social sentiment for {symbol}: {str(e)}"}
            )
            return []
