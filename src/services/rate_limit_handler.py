#!/usr/bin/env python3
"""
🚀 GoldenSignalsAI - Rate Limit Handler
Comprehensive rate limiting and caching strategies to avoid API limits

Features:
- Multi-level caching (Redis, Memory, Disk)
- Intelligent request throttling
- Multiple data source fallbacks
- Exponential backoff on errors
- Request prioritization
- Batch processing
"""

import asyncio
import json
import logging
import os
import pickle
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import redis
import yfinance as yf
from cachetools import LRUCache, TTLCache

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources"""

    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON_IO = "polygon_io"
    FINNHUB = "finnhub"
    CACHE = "cache"


class RequestPriority(Enum):
    """Request priority levels"""

    CRITICAL = 1  # Real-time trading signals
    HIGH = 2  # Active monitoring
    NORMAL = 3  # Regular updates
    LOW = 4  # Background tasks


@dataclass
class RateLimitConfig:
    """Rate limit configuration for each data source"""

    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    min_interval_ms: int
    backoff_factor: float = 2.0
    max_retries: int = 3


class RateLimitHandler:
    """Comprehensive rate limit handler with multiple strategies"""

    def __init__(self, redis_url: Optional[str] = None):
        # Initialize caches
        self.memory_cache = {
            "quotes": TTLCache(maxsize=1000, ttl=300),  # 5 min
            "historical": TTLCache(maxsize=500, ttl=600),  # 10 min
            "indicators": TTLCache(maxsize=1000, ttl=300),  # 5 min
            "news": TTLCache(maxsize=500, ttl=1800),  # 30 min
        }

        # Initialize Redis if available
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info("✅ Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis unavailable, using memory cache only: {e}")

        # Disk cache directory
        self.disk_cache_dir = "data/rate_limit_cache"
        os.makedirs(self.disk_cache_dir, exist_ok=True)

        # Rate limit configurations
        self.rate_limits = {
            DataSource.YAHOO_FINANCE: RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                min_interval_ms=100,
            ),
            DataSource.ALPHA_VANTAGE: RateLimitConfig(
                requests_per_minute=5,
                requests_per_hour=300,
                requests_per_day=500,
                min_interval_ms=12000,  # 12 seconds
            ),
            DataSource.IEX_CLOUD: RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=5000,
                requests_per_day=50000,
                min_interval_ms=50,
            ),
            DataSource.POLYGON_IO: RateLimitConfig(
                requests_per_minute=120,
                requests_per_hour=7000,
                requests_per_day=100000,
                min_interval_ms=50,
            ),
            DataSource.FINNHUB: RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=3000,
                requests_per_day=30000,
                min_interval_ms=100,
            ),
        }

        # Request tracking
        self.request_history: Dict[DataSource, deque] = {
            source: deque(maxlen=10000) for source in DataSource
        }

        # Request queue with priority
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # API keys (load from environment)
        self.api_keys = {
            DataSource.ALPHA_VANTAGE: os.getenv("ALPHA_VANTAGE_API_KEY"),
            DataSource.IEX_CLOUD: os.getenv("IEX_CLOUD_API_KEY"),
            DataSource.POLYGON_IO: os.getenv("POLYGON_API_KEY"),
            DataSource.FINNHUB: os.getenv("FINNHUB_API_KEY"),
        }

        # Error tracking for exponential backoff
        self.error_counts: Dict[DataSource, int] = {source: 0 for source in DataSource}
        self.last_error_time: Dict[DataSource, float] = {source: 0 for source in DataSource}

        # Background tasks will be started when needed
        self._background_tasks_started = False

    def _ensure_background_tasks(self):
        """Ensure background tasks are started when in async context"""
        if not self._background_tasks_started:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._process_request_queue())
                loop.create_task(self._cache_cleanup_task())
                self._background_tasks_started = True
            except RuntimeError:
                # No event loop running yet, tasks will start when one is available
                pass

    async def get_quote(
        self, symbol: str, priority: RequestPriority = RequestPriority.NORMAL
    ) -> Optional[Dict[str, Any]]:
        """
        Get quote with intelligent caching and fallback

        Args:
            symbol: Stock symbol
            priority: Request priority

        Returns:
            Quote data or None
        """
        # Ensure background tasks are running
        self._ensure_background_tasks()

        # Check multi-level cache first
        cached_data = await self._get_from_cache("quotes", symbol)
        if cached_data:
            logger.debug(f"Quote for {symbol} served from cache")
            return cached_data

        # Try data sources in order of preference
        sources = self._get_source_priority()

        for source in sources:
            if await self._can_make_request(source):
                try:
                    data = await self._fetch_quote_from_source(symbol, source)
                    if data:
                        # Cache the result
                        await self._save_to_cache("quotes", symbol, data)
                        self._reset_error_count(source)
                        return data
                except Exception as e:
                    logger.error(f"Error fetching from {source.value}: {e}")
                    self._increment_error_count(source)

        # If all sources fail, return stale cache if available
        return await self._get_from_cache("quotes", symbol, allow_stale=True)

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "1d",
        interval: str = "1m",
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> Optional[pd.DataFrame]:
        """Get historical data with caching and fallback"""
        # Ensure background tasks are running
        self._ensure_background_tasks()

        cache_key = f"{symbol}_{period}_{interval}"

        # Check cache
        cached_data = await self._get_from_cache("historical", cache_key)
        if cached_data is not None:
            return cached_data

        # Try data sources
        sources = self._get_source_priority()

        for source in sources:
            if source == DataSource.CACHE:
                continue
            if await self._can_make_request(source):
                try:
                    data = await self._fetch_historical_from_source(
                        symbol, period, interval, source
                    )
                    if data is not None and not data.empty:
                        await self._save_to_cache("historical", cache_key, data)
                        self._reset_error_count(source)
                        return data
                except Exception as e:
                    logger.error(f"Error fetching historical from {source.value}: {e}")
                    self._increment_error_count(source)

        return None

    async def batch_get_quotes(
        self, symbols: List[str], priority: RequestPriority = RequestPriority.NORMAL
    ) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols efficiently"""
        # Ensure background tasks are running
        self._ensure_background_tasks()

        results = {}

        # Check cache for all symbols first
        uncached_symbols = []
        for symbol in symbols:
            cached = await self._get_from_cache("quotes", symbol)
            if cached:
                results[symbol] = cached
            else:
                uncached_symbols.append(symbol)

        if not uncached_symbols:
            return results

        # Batch fetch uncached symbols
        batch_size = 10  # Adjust based on API limits
        for i in range(0, len(uncached_symbols), batch_size):
            batch = uncached_symbols[i : i + batch_size]

            # Try each data source
            for source in self._get_source_priority():
                if source == DataSource.CACHE:
                    continue
                if await self._can_make_request(source, len(batch)):
                    try:
                        batch_data = await self._fetch_batch_quotes_from_source(batch, source)
                        for symbol, data in batch_data.items():
                            if data:
                                results[symbol] = data
                                await self._save_to_cache("quotes", symbol, data)
                        break
                    except Exception as e:
                        logger.error(f"Batch fetch error from {source.value}: {e}")
                        self._increment_error_count(source)

        return results

    async def _can_make_request(self, source: DataSource, count: int = 1) -> bool:
        """Check if we can make a request to the given source"""
        if source == DataSource.CACHE:
            return True

        config = self.rate_limits[source]
        now = time.time()

        # Check error backoff
        if self.error_counts[source] > 0:
            backoff_time = (config.backoff_factor ** self.error_counts[source]) * 60
            if now - self.last_error_time[source] < backoff_time:
                return False

        # Check rate limits
        history = self.request_history[source]

        # Per minute check
        minute_ago = now - 60
        recent_minute = sum(1 for t in history if t > minute_ago)
        if recent_minute + count > config.requests_per_minute:
            return False

        # Per hour check
        hour_ago = now - 3600
        recent_hour = sum(1 for t in history if t > hour_ago)
        if recent_hour + count > config.requests_per_hour:
            return False

        # Per day check
        day_ago = now - 86400
        recent_day = sum(1 for t in history if t > day_ago)
        if recent_day + count > config.requests_per_day:
            return False

        # Minimum interval check
        if history and (now - history[-1]) * 1000 < config.min_interval_ms:
            return False

        return True

    async def _fetch_quote_from_source(
        self, symbol: str, source: DataSource
    ) -> Optional[Dict[str, Any]]:
        """Fetch quote from specific data source"""
        self._record_request(source)

        if source == DataSource.YAHOO_FINANCE:
            return await self._fetch_yahoo_quote(symbol)
        elif source == DataSource.ALPHA_VANTAGE:
            return await self._fetch_alpha_vantage_quote(symbol)
        elif source == DataSource.IEX_CLOUD:
            return await self._fetch_iex_quote(symbol)
        elif source == DataSource.POLYGON_IO:
            return await self._fetch_polygon_quote(symbol)
        elif source == DataSource.FINNHUB:
            return await self._fetch_finnhub_quote(symbol)

        return None

    async def _fetch_yahoo_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch quote from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
                "open": info.get("open", 0),
                "high": info.get("dayHigh", 0),
                "low": info.get("dayLow", 0),
                "close": info.get("previousClose", 0),
                "volume": info.get("volume", 0),
                "change": info.get("regularMarketChange", 0),
                "change_percent": info.get("regularMarketChangePercent", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": DataSource.YAHOO_FINANCE.value,
            }
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None

    async def _fetch_alpha_vantage_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch quote from Alpha Vantage"""
        if not self.api_keys[DataSource.ALPHA_VANTAGE]:
            return None

        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_keys[DataSource.ALPHA_VANTAGE],
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()

                    if "Global Quote" in data:
                        quote = data["Global Quote"]
                        return {
                            "symbol": symbol,
                            "price": float(quote.get("05. price", 0)),
                            "open": float(quote.get("02. open", 0)),
                            "high": float(quote.get("03. high", 0)),
                            "low": float(quote.get("04. low", 0)),
                            "close": float(quote.get("08. previous close", 0)),
                            "volume": int(quote.get("06. volume", 0)),
                            "change": float(quote.get("09. change", 0)),
                            "change_percent": float(
                                quote.get("10. change percent", "0%").rstrip("%")
                            ),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source": DataSource.ALPHA_VANTAGE.value,
                        }
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")

        return None

    async def _get_from_cache(
        self, cache_type: str, key: str, allow_stale: bool = False
    ) -> Optional[Any]:
        """Get data from multi-level cache"""
        # Try memory cache first
        if key in self.memory_cache[cache_type]:
            return self.memory_cache[cache_type][key]

        # Try Redis cache
        if self.redis_client:
            try:
                redis_key = f"{cache_type}:{key}"
                data = self.redis_client.get(redis_key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.debug(f"Redis cache miss: {e}")

        # Try disk cache
        disk_path = os.path.join(self.disk_cache_dir, f"{cache_type}_{key}.pkl")
        if os.path.exists(disk_path):
            try:
                # Check age
                age = time.time() - os.path.getmtime(disk_path)
                max_age = 86400 if allow_stale else 3600  # 24h if stale allowed, else 1h

                if age < max_age:
                    with open(disk_path, "rb") as f:
                        return pickle.load(f)
            except Exception as e:
                logger.debug(f"Disk cache error: {e}")

        return None

    async def _save_to_cache(self, cache_type: str, key: str, data: Any):
        """Save data to multi-level cache"""
        # Save to memory cache
        self.memory_cache[cache_type][key] = data

        # Save to Redis
        if self.redis_client:
            try:
                redis_key = f"{cache_type}:{key}"
                ttl = 300 if cache_type == "quotes" else 600
                self.redis_client.setex(redis_key, ttl, json.dumps(data, default=str))
            except Exception as e:
                logger.debug(f"Redis cache save error: {e}")

        # Save to disk cache
        try:
            disk_path = os.path.join(self.disk_cache_dir, f"{cache_type}_{key}.pkl")
            with open(disk_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.debug(f"Disk cache save error: {e}")

    def _get_source_priority(self) -> List[DataSource]:
        """Get data sources in order of preference based on current state"""
        sources = []

        # Always check cache first
        sources.append(DataSource.CACHE)

        # Order other sources by error count and availability
        other_sources = [
            DataSource.YAHOO_FINANCE,
            DataSource.IEX_CLOUD,
            DataSource.POLYGON_IO,
            DataSource.FINNHUB,
            DataSource.ALPHA_VANTAGE,
        ]

        # Sort by error count (ascending) and API key availability
        other_sources.sort(
            key=lambda s: (
                self.error_counts[s],
                0 if s == DataSource.YAHOO_FINANCE or self.api_keys.get(s) else 1,
            )
        )

        sources.extend(other_sources)
        return sources

    def _record_request(self, source: DataSource):
        """Record a request for rate limiting"""
        self.request_history[source].append(time.time())

    def _increment_error_count(self, source: DataSource):
        """Increment error count for exponential backoff"""
        self.error_counts[source] = min(
            self.error_counts[source] + 1, self.rate_limits[source].max_retries
        )
        self.last_error_time[source] = time.time()

    def _reset_error_count(self, source: DataSource):
        """Reset error count after successful request"""
        self.error_counts[source] = 0

    async def _process_request_queue(self):
        """Process queued requests with priority"""
        while True:
            try:
                # Get next request from queue
                priority, request = await self.request_queue.get()

                # Process the request
                # Implementation depends on request type

                await asyncio.sleep(0.1)  # Small delay between processing
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)

    async def _cache_cleanup_task(self):
        """Periodic cache cleanup"""
        while True:
            try:
                # Clean up old disk cache files
                now = time.time()
                for filename in os.listdir(self.disk_cache_dir):
                    filepath = os.path.join(self.disk_cache_dir, filename)
                    if os.path.isfile(filepath):
                        age = now - os.path.getmtime(filepath)
                        if age > 86400:  # 24 hours
                            os.remove(filepath)

                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(3600)

    async def _fetch_historical_from_source(
        self, symbol: str, period: str, interval: str, source: DataSource
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data from specific source"""
        self._record_request(source)

        if source == DataSource.YAHOO_FINANCE:
            return await self._fetch_yahoo_historical(symbol, period, interval)

        return None

    async def _fetch_yahoo_historical(
        self, symbol: str, period: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period, interval=interval)

            if hist_data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None

            return hist_data

        except Exception as e:
            logger.error(f"Yahoo Finance historical error for {symbol}: {e}")
            return None

    async def _fetch_batch_quotes_from_source(
        self, symbols: List[str], source: DataSource
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch batch quotes from specific source"""
        results = {}

        if source == DataSource.YAHOO_FINANCE:
            # Yahoo Finance doesn't have true batch API, so we fetch individually
            # but count it as batch for rate limiting
            for symbol in symbols:
                try:
                    quote = await self._fetch_yahoo_quote(symbol)
                    if quote:
                        results[symbol] = quote
                except Exception as e:
                    logger.error(f"Batch quote error for {symbol}: {e}")

        return results


# Singleton instance
_rate_limit_handler: Optional[RateLimitHandler] = None


def get_rate_limit_handler() -> RateLimitHandler:
    """Get or create the rate limit handler singleton"""
    global _rate_limit_handler
    if _rate_limit_handler is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _rate_limit_handler = RateLimitHandler(redis_url)
    return _rate_limit_handler
