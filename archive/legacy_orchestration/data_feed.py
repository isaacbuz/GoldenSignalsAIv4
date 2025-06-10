# orchestration/data_feed.py
# Purpose: Streams real-time market data using Redis Streams and archives to TimescaleDB.
# Enhanced for options trading with options data streaming and multi-source integration
# (stock, options, news, social media).

import asyncio
import logging

import pandas as pd
import redis
import yaml

from infrastructure.data_fetcher import DataFetcher

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class RealTimeDataFeed:
    """Streams real-time market data using Redis Streams."""

    def __init__(self, symbols: list, interval: str = "1d", period: str = "1mo"):
        """Initialize the RealTimeDataFeed.

        Args:
            symbols (list): List of stock symbols.
            interval (str): Data interval ('1m', '1d', etc.).
            period (str): Data period ('1d', '1mo', etc.).
        """
        self.symbols = symbols
        self.interval = interval
        self.period = period
        self.fetcher = DataFetcher()
        # Initialize Redis client
        if config["redis"].get("cluster_enabled", False):
            from redis.cluster import RedisCluster

            nodes = config["redis"]["cluster_nodes"]
            self.redis_client = RedisCluster(
                startup_nodes=[
                    {"host": node["host"], "port": node["port"]} for node in nodes
                ]
            )
        else:
            self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.historical_data = pd.DataFrame()
        logger.info({"message": f"RealTimeDataFeed initialized for symbols: {symbols}"})

    async def __aiter__(self):
        """Async iterator for streaming real-time data.

        Yields:
            dict: Market observation for each symbol.
        """
        logger.info({"message": "Starting real-time data stream"})
        while True:
            try:
                for symbol in self.symbols:
                    # Fetch stock, options, news, and social media data
                    stock_data = self.fetcher.fetch_stock_data(
                        symbol, self.interval, self.period
                    )
                    options_data = self.fetcher.fetch_options_data(symbol)
                    news_articles = self.fetcher.fetch_news_data(symbol)
                    social_media = self.fetcher.fetch_social_sentiment(symbol)
                    # Update historical data
                    self.historical_data = pd.concat(
                        [self.historical_data, stock_data], ignore_index=True
                    )
                    # Construct observation
                    observation = {
                        "symbol": symbol,
                        "stock_data": stock_data,
                        "options_data": options_data,
                        "news_articles": news_articles,
                        "social_media": social_media,
                        "prices": {
                            symbol: stock_data["Close"].iloc[-1]
                            if not stock_data.empty
                            else 0.0
                        },
                    }
                    # Publish to Redis Stream
                    stream_data = {
                        "symbol": symbol,
                        "stock_data": stock_data.to_json(),
                        "options_data": options_data.to_json(),
                        "news_articles": str(news_articles),
                        "social_media": str(social_media),
                    }
                    self.redis_client.xadd("market-data-stream", stream_data)
                    logger.info({"message": f"Streamed observation for {symbol}"})
                    yield observation
                await asyncio.sleep(60)  # Fetch every minute
            except Exception as e:
                logger.error({"message": f"Failed to fetch data: {str(e)}"})
                await asyncio.sleep(60)
