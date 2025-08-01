import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import finnhub
import newsapi
import pandas as pd
import polygon
import tweepy
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from benzinga import news_data

from config.settings import settings

logger = logging.getLogger(__name__)


class MultiSourceDataFetcher:
    """Fetches data from multiple sources with fallback options"""

    def __init__(self):
        """Initialize data fetchers for all sources"""
        # Market Data Sources
        self.alpha_vantage = TimeSeries(key=settings.ALPHA_VANTAGE_API_KEY, output_format="pandas")
        self.finnhub_client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)
        self.polygon_client = polygon.RESTClient(settings.POLYGON_API_KEY)

        # News Sources
        self.newsapi_client = newsapi.NewsApiClient(api_key=settings.NEWS_API_KEY)
        self.benzinga_client = news_data.News(api_key=settings.BENZINGA_API_KEY)

        # Twitter
        self.twitter_client = tweepy.Client(bearer_token=settings.TWITTER_API_KEY)

    def fetch_market_data(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetch market data with fallback options"""
        # Try Alpha Vantage first
        try:
            df = self._fetch_alpha_vantage_data(symbol, start_date, end_date, interval)
            if df is not None:
                return df
        except Exception as e:
            logger.warning(f"Alpha Vantage failed for {symbol}: {str(e)}")

        # Try Finnhub
        try:
            df = self._fetch_finnhub_data(symbol, start_date, end_date, interval)
            if df is not None:
                return df
        except Exception as e:
            logger.warning(f"Finnhub failed for {symbol}: {str(e)}")

        # Try Polygon
        try:
            df = self._fetch_polygon_data(symbol, start_date, end_date, interval)
            if df is not None:
                return df
        except Exception as e:
            logger.warning(f"Polygon failed for {symbol}: {str(e)}")

        # Try Yahoo Finance as last resort
        try:
            df = self._fetch_yahoo_data(symbol, start_date, end_date, interval)
            if df is not None:
                return df
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {symbol}: {str(e)}")

        logger.error(f"Failed to fetch market data for {symbol} from all sources")
        return None

    def fetch_news(
        self, symbol: str, start_date: datetime, end_date: datetime, limit: int = 100
    ) -> List[Dict]:
        """Fetch news from multiple sources"""
        news_items = []

        # Try News API
        try:
            news_items.extend(self._fetch_newsapi_news(symbol, start_date, end_date, limit))
        except Exception as e:
            logger.warning(f"News API failed for {symbol}: {str(e)}")

        # Try Benzinga
        try:
            news_items.extend(self._fetch_benzinga_news(symbol, start_date, end_date, limit))
        except Exception as e:
            logger.warning(f"Benzinga failed for {symbol}: {str(e)}")

        return news_items

    def fetch_social_sentiment(
        self, symbol: str, start_date: datetime, end_date: datetime, limit: int = 100
    ) -> List[Dict]:
        """Fetch social media sentiment"""
        tweets = []

        # Try Twitter
        try:
            tweets.extend(self._fetch_twitter_sentiment(symbol, start_date, end_date, limit))
        except Exception as e:
            logger.warning(f"Twitter failed for {symbol}: {str(e)}")

        return tweets

    def _fetch_alpha_vantage_data(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str
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
                return None

            # Rename columns to match standard format
            df = df.rename(
                columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",
                    "5. volume": "Volume",
                }
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {str(e)}")
            return None

    def _fetch_finnhub_data(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Finnhub"""
        try:
            # Convert dates to Unix timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())

            # Fetch data
            data = self.finnhub_client.stock_candles(
                symbol, interval, start_timestamp, end_timestamp
            )

            if not data or data["s"] != "ok":
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                {
                    "Open": data["o"],
                    "High": data["h"],
                    "Low": data["l"],
                    "Close": data["c"],
                    "Volume": data["v"],
                },
                index=pd.to_datetime(data["t"], unit="s"),
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching Finnhub data: {str(e)}")
            return None

    def _fetch_polygon_data(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon"""
        try:
            # Convert interval to Polygon format
            interval_map = {"1d": "day", "1h": "hour", "5m": "minute"}
            poly_interval = interval_map.get(interval, "day")

            # Fetch data
            data = self.polygon_client.get_aggs(
                symbol, multiplier=1, timespan=poly_interval, from_=start_date, to=end_date
            )

            if not data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "Open": bar.open,
                        "High": bar.high,
                        "Low": bar.low,
                        "Close": bar.close,
                        "Volume": bar.volume,
                    }
                    for bar in data
                ]
            )

            df.index = pd.to_datetime([bar.timestamp for bar in data])
            return df

        except Exception as e:
            logger.error(f"Error fetching Polygon data: {str(e)}")
            return None

    def _fetch_yahoo_data(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                return None

            return df

        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {str(e)}")
            return None

    def _fetch_newsapi_news(
        self, symbol: str, start_date: datetime, end_date: datetime, limit: int
    ) -> List[Dict]:
        """Fetch news from News API"""
        try:
            # Format dates
            from_date = start_date.strftime("%Y-%m-%d")
            to_date = end_date.strftime("%Y-%m-%d")

            # Fetch news
            response = self.newsapi_client.get_everything(
                q=symbol,
                from_param=from_date,
                to=to_date,
                language="en",
                sort_by="relevancy",
                page_size=limit,
            )

            return response["articles"]

        except Exception as e:
            logger.error(f"Error fetching News API data: {str(e)}")
            return []

    def _fetch_benzinga_news(
        self, symbol: str, start_date: datetime, end_date: datetime, limit: int
    ) -> List[Dict]:
        """Fetch news from Benzinga"""
        try:
            # Fetch news
            news = self.benzinga_client.news(
                tickers=symbol, date_from=start_date, date_to=end_date, limit=limit
            )

            return news

        except Exception as e:
            logger.error(f"Error fetching Benzinga data: {str(e)}")
            return []

    def _fetch_twitter_sentiment(
        self, symbol: str, start_date: datetime, end_date: datetime, limit: int
    ) -> List[Dict]:
        """Fetch Twitter sentiment"""
        try:
            # Format query
            query = f"${symbol} -is:retweet lang:en"

            # Fetch tweets
            tweets = self.twitter_client.search_recent_tweets(
                query=query, start_time=start_date, end_time=end_date, max_results=limit
            )

            return tweets.data if tweets.data else []

        except Exception as e:
            logger.error(f"Error fetching Twitter data: {str(e)}")
            return []
