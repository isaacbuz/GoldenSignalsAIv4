"""
Data source agents for fetching market data from various providers.
"""

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


# Base Agent Interface
class DataSourceAgent:
    def fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    def fetch_news(self, symbol: str) -> List[Dict]:
        raise NotImplementedError

    def fetch_sentiment(self, symbol: str) -> List[Dict]:
        raise NotImplementedError


# Alpha Vantage Agent
class AlphaVantageAgent(DataSourceAgent):
    """
    Agent for Alpha Vantage API. Handles missing or expired API key gracefully.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.api_key:
            logger.warning({"message": "Alpha Vantage API key missing."})
            return None
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}"
            response = requests.get(url)
            if response.status_code == 403 or response.status_code == 401:
                logger.error({"message": f"Alpha Vantage API key expired or invalid for {symbol}."})
                return None
            response.raise_for_status()
            data = response.json().get("Time Series (Daily)", {})
            if not data:
                logger.warning({"message": f"Alpha Vantage: No data for {symbol}"})
                return None
            df = pd.DataFrame.from_dict(data, orient="index").astype(float)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.error({"message": f"Alpha Vantage error for {symbol}: {str(e)}"})
            return None

    def fetch_news(self, symbol: str) -> List[Dict]:
        return []

    def fetch_sentiment(self, symbol: str) -> List[Dict]:
        return []


# Finnhub Agent
class FinnhubAgent(DataSourceAgent):
    """
    Agent for Finnhub API. Handles missing or expired API key gracefully.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.api_key:
            logger.warning({"message": "Finnhub API key missing."})
            return None
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.api_key}"
            response = requests.get(url)
            if response.status_code == 403 or response.status_code == 401:
                logger.error({"message": f"Finnhub API key expired or invalid for {symbol}."})
                return None
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame([{**data, "symbol": symbol}])
            return df
        except Exception as e:
            logger.error({"message": f"Finnhub error for {symbol}: {str(e)}"})
            return None

    def fetch_news(self, symbol: str) -> List[Dict]:
        if not self.api_key:
            logger.warning({"message": "Finnhub API key missing."})
            return []
        try:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2025-05-01&to=2025-05-14&token={self.api_key}"
            response = requests.get(url)
            if response.status_code == 403 or response.status_code == 401:
                logger.error({"message": f"Finnhub API key expired or invalid for {symbol}."})
                return []
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error({"message": f"Finnhub news error for {symbol}: {str(e)}"})
            return []

    def fetch_sentiment(self, symbol: str) -> List[Dict]:
        return []


# Polygon Agent
class PolygonAgent(DataSourceAgent):
    """
    Agent for Polygon.io API. Handles missing or expired API key gracefully.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.api_key:
            logger.warning({"message": "Polygon API key missing."})
            return None
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={self.api_key}"
            response = requests.get(url)
            if response.status_code == 403 or response.status_code == 401:
                logger.error({"message": f"Polygon API key expired or invalid for {symbol}."})
                return None
            response.raise_for_status()
            results = response.json().get("results", [])
            if not results:
                logger.warning({"message": f"Polygon: No data for {symbol}"})
                return None
            df = pd.DataFrame(results)
            return df
        except Exception as e:
            logger.error({"message": f"Polygon error for {symbol}: {str(e)}"})
            return None

    def fetch_news(self, symbol: str) -> List[Dict]:
        return []

    def fetch_sentiment(self, symbol: str) -> List[Dict]:
        return []


# Benzinga News Agent
class BenzingaNewsAgent(DataSourceAgent):
    """
    BenzingaNewsAgent fetches news articles from Benzinga's API. Handles missing or expired API key gracefully.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return None

    def fetch_news(self, symbol: str) -> List[Dict]:
        if not self.api_key:
            logger.warning({"message": "Benzinga API key missing."})
            return []
        try:
            url = f"https://api.benzinga.com/api/v2/news?token={self.api_key}&symbols={symbol}"
            response = requests.get(url)
            if response.status_code == 403 or response.status_code == 401:
                logger.error({"message": f"Benzinga API key expired or invalid for {symbol}."})
                return []
            response.raise_for_status()
            return response.json().get("articles", [])
        except Exception as e:
            logger.error({"message": f"Benzinga news error for {symbol}: {str(e)}"})
            return []

    def fetch_sentiment(self, symbol: str) -> List[Dict]:
        return []


# StockTwits Sentiment Agent
class StockTwitsAgent(DataSourceAgent):
    def fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return None

    def fetch_news(self, symbol: str) -> List[Dict]:
        return []

    def fetch_sentiment(self, symbol: str) -> List[Dict]:
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            response = requests.get(url)
            response.raise_for_status()
            messages = response.json().get("messages", [])
            sentiments = [
                {
                    "text": m["body"],
                    "sentiment": m["entities"].get("sentiment", {}).get("basic", "neutral"),
                }
                for m in messages
            ]
            return sentiments
        except Exception as e:
            logger.error({"message": f"StockTwits sentiment error for {symbol}: {str(e)}"})
            return []


# Bloomberg Agent
class BloombergAgent(DataSourceAgent):
    """
    BloombergAgent integrates with Bloomberg's blpapi for price and news data. Handles missing or expired API key gracefully.
    Requires Bloomberg Terminal and blpapi installed.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key  # Placeholder, Bloomberg uses Terminal login not API key
        try:
            import blpapi

            self.blpapi = blpapi
        except ImportError:
            self.blpapi = None
            logger.warning({"message": "blpapi not installed; BloombergAgent will not function."})

    def fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.blpapi:
            logger.error({"message": "blpapi not available"})
            return None
        if self.api_key is not None and self.api_key.strip() == "":
            logger.warning({"message": "Bloomberg API key missing."})
            return None
        # Placeholder: Actual Bloomberg Terminal integration required
        logger.info({"message": f"BloombergAgent fetch_price_data for {symbol} (stub)"})
        return None

    def fetch_news(self, symbol: str) -> List[Dict]:
        if not self.blpapi:
            logger.error({"message": "blpapi not available"})
            return []
        if self.api_key is not None and self.api_key.strip() == "":
            logger.warning({"message": "Bloomberg API key missing."})
            return []
        logger.info({"message": f"BloombergAgent fetch_news for {symbol} (stub)"})
        return []

    def fetch_sentiment(self, symbol: str) -> List[Dict]:
        return []


# Data Aggregator Agent
class DataAggregator:
    def __init__(self, agents: List[DataSourceAgent]):
        self.agents = agents

    def fetch_all_price_data(self, symbol: str) -> List[pd.DataFrame]:
        return [
            agent.fetch_price_data(symbol)
            for agent in self.agents
            if hasattr(agent, "fetch_price_data")
        ]

    def fetch_all_news(self, symbol: str) -> List[List[Dict]]:
        return [agent.fetch_news(symbol) for agent in self.agents if hasattr(agent, "fetch_news")]

    def fetch_all_sentiment(self, symbol: str) -> List[List[Dict]]:
        return [
            agent.fetch_sentiment(symbol)
            for agent in self.agents
            if hasattr(agent, "fetch_sentiment")
        ]


# Example initialization (would be done in your service layer)
def get_default_data_aggregator():
    agents = []
    if os.getenv("ALPHA_VANTAGE_API_KEY"):
        agents.append(AlphaVantageAgent(os.getenv("ALPHA_VANTAGE_API_KEY")))
    if os.getenv("FINNHUB_API_KEY"):
        agents.append(FinnhubAgent(os.getenv("FINNHUB_API_KEY")))
    if os.getenv("POLYGON_API_KEY"):
        agents.append(PolygonAgent(os.getenv("POLYGON_API_KEY")))
    if os.getenv("BENZINGA_API_KEY"):
        agents.append(BenzingaNewsAgent(os.getenv("BENZINGA_API_KEY")))
    if (
        os.getenv("BLPAPI_KEY") or True
    ):  # Always add BloombergAgent for support, even if key is missing
        agents.append(BloombergAgent(os.getenv("BLPAPI_KEY")))
    agents.append(StockTwitsAgent())
    return DataAggregator(agents)
