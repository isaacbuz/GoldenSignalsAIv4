#!/usr/bin/env python
"""
Market Data Caching Service
Implements Redis-based caching to prevent API rate limiting
"""

import json
import redis
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class MarketDataCache:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=1):
        """Initialize Redis connection for caching"""
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache: {e}")
            self.redis_client = None
            self.memory_cache = {}

    def _get_cache_key(self, symbol: str, period: str, interval: str) -> str:
        """Generate cache key for market data"""
        return f"market_data:{symbol}:{period}:{interval}"

    def _get_ttl(self, interval: str) -> int:
        """Get cache TTL based on interval"""
        ttl_map = {
            '1m': 60,        # 1 minute
            '5m': 300,       # 5 minutes
            '15m': 900,      # 15 minutes
            '30m': 1800,     # 30 minutes
            '1h': 3600,      # 1 hour
            '1d': 86400,     # 1 day
            '1wk': 604800,   # 1 week
            '1mo': 2592000,  # 30 days
            '3mo': 7776000,  # 90 days
        }
        return ttl_map.get(interval, 3600)  # Default 1 hour

    def get_cached_data(self, symbol: str, period: str, interval: str) -> Optional[List[Dict]]:
        """Get data from cache if available"""
        cache_key = self._get_cache_key(symbol, period, interval)

        try:
            if self.redis_client:
                cached = self.redis_client.get(cache_key)
                if cached:
                    logger.info(f"Cache hit for {symbol} {period} {interval}")
                    return json.loads(cached)
            else:
                # In-memory fallback
                if cache_key in self.memory_cache:
                    cached_data, expiry = self.memory_cache[cache_key]
                    if datetime.now() < expiry:
                        logger.info(f"Memory cache hit for {symbol}")
                        return cached_data
                    else:
                        del self.memory_cache[cache_key]
        except Exception as e:
            logger.error(f"Cache get error: {e}")

        return None

    def set_cached_data(self, symbol: str, period: str, interval: str, data: List[Dict]) -> bool:
        """Store data in cache"""
        cache_key = self._get_cache_key(symbol, period, interval)
        ttl = self._get_ttl(interval)

        try:
            if self.redis_client:
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(data)
                )
                logger.info(f"Cached {len(data)} candles for {symbol} (TTL: {ttl}s)")
                return True
            else:
                # In-memory fallback
                expiry = datetime.now() + timedelta(seconds=ttl)
                self.memory_cache[cache_key] = (data, expiry)
                # Limit memory cache size
                if len(self.memory_cache) > 100:
                    oldest_key = min(self.memory_cache.keys(),
                                   key=lambda k: self.memory_cache[k][1])
                    del self.memory_cache[oldest_key]
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def clear_symbol_cache(self, symbol: str) -> int:
        """Clear all cached data for a symbol"""
        cleared = 0
        try:
            if self.redis_client:
                pattern = f"market_data:{symbol}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    cleared = self.redis_client.delete(*keys)
            else:
                # In-memory clear
                keys_to_delete = [k for k in self.memory_cache.keys()
                                if k.startswith(f"market_data:{symbol}:")]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                    cleared += 1

            logger.info(f"Cleared {cleared} cache entries for {symbol}")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

        return cleared

# Global cache instance
market_cache = MarketDataCache()

def get_market_data_with_cache(symbol: str, period: str, interval: str) -> List[Dict]:
    """Get market data with caching to prevent rate limits"""

    # Check cache first
    cached_data = market_cache.get_cached_data(symbol, period, interval)
    if cached_data:
        return cached_data

    # Fetch fresh data
    try:
        logger.info(f"Fetching fresh data for {symbol} {period} {interval}")

        # Clean symbol for yfinance
        yf_symbol = symbol.replace("-USD", "-USD") if "-USD" in symbol else symbol
        ticker = yf.Ticker(yf_symbol)

        # Fetch data
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return []

        # Convert to list of dicts
        candles = []
        for index, row in df.iterrows():
            candles.append({
                "time": int(index.timestamp()),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "volume": int(row['Volume'])
            })

        # Cache the data
        market_cache.set_cached_data(symbol, period, interval, candles)

        return candles

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return []

def get_fallback_providers(symbol: str, period: str, interval: str) -> List[Dict]:
    """Try alternative data providers when primary fails"""

    providers = [
        # Add other providers here as needed
        # Example: alpha_vantage, finnhub, polygon, etc.
    ]

    for provider in providers:
        try:
            # Implementation for each provider
            pass
        except Exception as e:
            logger.warning(f"Provider {provider} failed: {e}")
            continue

    # If all fail, generate synthetic data as last resort
    return generate_synthetic_data(symbol, period, interval)

def generate_synthetic_data(symbol: str, period: str, interval: str) -> List[Dict]:
    """Generate synthetic data when all providers fail"""
    import numpy as np

    # Determine number of candles based on period/interval
    candle_counts = {
        ('1d', '1m'): 390,    # Trading day minutes
        ('5d', '5m'): 390,    # 5 days of 5m
        ('1mo', '1h'): 160,   # ~20 trading days
        ('1y', '1d'): 252,    # Trading days in year
        ('max', '1mo'): 120,  # 10 years monthly
    }

    count = candle_counts.get((period, interval), 100)

    # Base prices for common symbols
    base_prices = {
        'AAPL': 175.0,
        'MSFT': 420.0,
        'GOOGL': 155.0,
        'BTC-USD': 68000.0,
        'ETH-USD': 3800.0,
        'SPY': 500.0,
    }

    base_price = base_prices.get(symbol, 100.0)

    # Generate realistic price movement
    candles = []
    current_time = int(datetime.now().timestamp())

    # Time intervals in seconds
    interval_seconds = {
        '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
        '1h': 3600, '1d': 86400, '1wk': 604800, '1mo': 2592000
    }.get(interval, 3600)

    for i in range(count):
        time_offset = (count - i - 1) * interval_seconds
        candle_time = current_time - time_offset

        # Random walk with trend
        volatility = 0.02
        trend = np.sin(i / count * 2 * np.pi) * 0.1
        change = np.random.normal(trend / count, volatility)

        base_price *= (1 + change)

        # Generate OHLC
        open_price = base_price
        close_price = base_price * (1 + np.random.uniform(-0.01, 0.01))
        high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.005))
        low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.005))
        volume = int(np.random.lognormal(15, 1))  # Log-normal volume

        candles.append({
            "time": candle_time,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume,
            "synthetic": True  # Mark as synthetic
        })

        base_price = close_price

    logger.info(f"Generated {len(candles)} synthetic candles for {symbol}")
    return candles

if __name__ == "__main__":
    # Test the cache
    logging.basicConfig(level=logging.INFO)

    # Test with AAPL
    data = get_market_data_with_cache("AAPL", "1d", "5m")
    print(f"Got {len(data)} candles")

    # Second call should hit cache
    data2 = get_market_data_with_cache("AAPL", "1d", "5m")
    print(f"Got {len(data2)} candles from cache")
