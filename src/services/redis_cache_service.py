"""
Redis Cache Service for AI Predictions
Implements caching layer to reduce API calls and improve response times
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import redis
from redis.exceptions import RedisError

from config.secure_config import ENVIRONMENT, REDIS_URL

logger = logging.getLogger(__name__)


class RedisCacheService:
    """
    Redis-based caching service for AI predictions and market data
    """

    def __init__(self, default_ttl: int = 60):
        """
        Initialize Redis cache service

        Args:
            default_ttl: Default time-to-live in seconds (60 seconds default)
        """
        self.default_ttl = default_ttl
        self.redis_client = None
        self.enabled = True
        self._connect()

    def _connect(self):
        """Establish Redis connection"""
        try:
            self.redis_client = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                retry_on_error=[ConnectionError, TimeoutError],
                max_connections=50,
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache service connected successfully")
        except (RedisError, Exception) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Redis caching disabled - falling back to direct API calls")
            self.enabled = False
            self.redis_client = None

    def _generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """
        Generate a consistent cache key from prefix and parameters

        Args:
            prefix: Key prefix (e.g., 'ai_prediction', 'market_data')
            params: Parameters to include in key

        Returns:
            Cache key string
        """
        # Sort params for consistent keys
        sorted_params = sorted(params.items())
        param_str = json.dumps(sorted_params, sort_keys=True)

        # Create hash for long parameter strings
        if len(param_str) > 100:
            param_hash = hashlib.md5(param_str.encode()).hexdigest()
            return f"{prefix}:{param_hash}"
        else:
            # Use readable key for short params
            param_parts = [f"{k}={v}" for k, v in sorted_params]
            return f"{prefix}:{':'.join(param_parts)}"

    async def get_ai_prediction(
        self, symbol: str, provider: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached AI prediction

        Args:
            symbol: Trading symbol
            provider: AI provider (openai, anthropic, etc.)
            context: Additional context for prediction

        Returns:
            Cached prediction or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            key = self._generate_key(
                "ai_prediction",
                {
                    "symbol": symbol,
                    "provider": provider,
                    "context_hash": hashlib.md5(
                        json.dumps(context or {}, sort_keys=True).encode()
                    ).hexdigest()[:8],
                },
            )

            cached_data = self.redis_client.get(key)
            if cached_data:
                prediction = json.loads(cached_data)

                # Add cache metadata
                prediction["_cached"] = True
                prediction["_cache_key"] = key

                logger.debug(f"Cache hit for AI prediction: {key}")
                return prediction

            logger.debug(f"Cache miss for AI prediction: {key}")
            return None

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set_ai_prediction(
        self,
        symbol: str,
        provider: str,
        prediction: Dict[str, Any],
        ttl: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Cache AI prediction

        Args:
            symbol: Trading symbol
            provider: AI provider
            prediction: Prediction data to cache
            ttl: Time-to-live in seconds (uses default if not specified)
            context: Additional context used for prediction

        Returns:
            Success status
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            key = self._generate_key(
                "ai_prediction",
                {
                    "symbol": symbol,
                    "provider": provider,
                    "context_hash": hashlib.md5(
                        json.dumps(context or {}, sort_keys=True).encode()
                    ).hexdigest()[:8],
                },
            )

            # Add timestamp to cached data
            cache_data = {
                **prediction,
                "_cached_at": datetime.now().isoformat(),
                "_ttl": ttl or self.default_ttl,
            }

            # Remove any existing cache metadata
            cache_data.pop("_cached", None)
            cache_data.pop("_cache_key", None)

            success = self.redis_client.setex(key, ttl or self.default_ttl, json.dumps(cache_data))

            if success:
                logger.debug(f"Cached AI prediction: {key} (TTL: {ttl or self.default_ttl}s)")

            return bool(success)

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def get_market_data(
        self, symbol: str, data_type: str, timeframe: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached market data

        Args:
            symbol: Trading symbol
            data_type: Type of data (price, volume, indicators, etc.)
            timeframe: Optional timeframe

        Returns:
            Cached data or None
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            params = {"symbol": symbol, "type": data_type}
            if timeframe:
                params["timeframe"] = timeframe

            key = self._generate_key("market_data", params)

            cached_data = self.redis_client.get(key)
            if cached_data:
                data = json.loads(cached_data)
                data["_cached"] = True
                logger.debug(f"Cache hit for market data: {key}")
                return data

            return None

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set_market_data(
        self,
        symbol: str,
        data_type: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        timeframe: Optional[str] = None,
    ) -> bool:
        """
        Cache market data

        Args:
            symbol: Trading symbol
            data_type: Type of data
            data: Data to cache
            ttl: Time-to-live in seconds
            timeframe: Optional timeframe

        Returns:
            Success status
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            params = {"symbol": symbol, "type": data_type}
            if timeframe:
                params["timeframe"] = timeframe

            key = self._generate_key("market_data", params)

            cache_data = {
                **data,
                "_cached_at": datetime.now().isoformat(),
                "_ttl": ttl or self.default_ttl,
            }

            # Market data typically has shorter TTL
            market_ttl = ttl or min(self.default_ttl, 30)

            success = self.redis_client.setex(key, market_ttl, json.dumps(cache_data))

            return bool(success)

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def get_agent_analysis(
        self, agent_name: str, symbol: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached agent analysis result
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            cache_params = {"agent": agent_name, "symbol": symbol, **(params or {})}

            key = self._generate_key("agent_analysis", cache_params)

            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)

            return None

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set_agent_analysis(
        self,
        agent_name: str,
        symbol: str,
        analysis: Dict[str, Any],
        ttl: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Cache agent analysis result
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            cache_params = {"agent": agent_name, "symbol": symbol, **(params or {})}

            key = self._generate_key("agent_analysis", cache_params)

            # Agent analysis can be cached slightly longer
            agent_ttl = ttl or min(self.default_ttl * 2, 120)

            success = self.redis_client.setex(key, agent_ttl, json.dumps(analysis))

            return bool(success)

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def invalidate_symbol_cache(self, symbol: str) -> int:
        """
        Invalidate all caches for a specific symbol

        Args:
            symbol: Trading symbol

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0

        try:
            pattern = f"*symbol={symbol}*"
            keys = self.redis_client.keys(pattern)

            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries for {symbol}")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Redis invalidate error: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Cache statistics including hit rate, memory usage, etc.
        """
        if not self.enabled or not self.redis_client:
            return {"enabled": False, "error": "Redis not connected"}

        try:
            info = self.redis_client.info()
            memory_info = self.redis_client.info("memory")

            # Get key counts by prefix
            ai_prediction_keys = len(self.redis_client.keys("ai_prediction:*"))
            market_data_keys = len(self.redis_client.keys("market_data:*"))
            agent_analysis_keys = len(self.redis_client.keys("agent_analysis:*"))

            return {
                "enabled": True,
                "connected": self.redis_client.ping(),
                "memory_used_mb": round(memory_info.get("used_memory", 0) / 1024 / 1024, 2),
                "memory_peak_mb": round(memory_info.get("used_memory_peak", 0) / 1024 / 1024, 2),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "key_counts": {
                    "ai_predictions": ai_prediction_keys,
                    "market_data": market_data_keys,
                    "agent_analysis": agent_analysis_keys,
                },
                "hit_rate": round(
                    info.get("keyspace_hits", 0)
                    / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1)
                    * 100,
                    2,
                ),
                "uptime_hours": round(info.get("uptime_in_seconds", 0) / 3600, 2),
            }

        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"enabled": True, "error": str(e)}

    async def clear_all_cache(self) -> bool:
        """
        Clear all cached data (use with caution)

        Returns:
            Success status
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            if ENVIRONMENT == "production":
                logger.warning("Attempted to clear all cache in production - denied")
                return False

            self.redis_client.flushdb()
            logger.info("Cleared all cache data")
            return True

        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False


# Global cache instance
redis_cache = RedisCacheService()


# Convenience decorators for caching
def cache_ai_prediction(ttl: int = 60):
    """
    Decorator to cache AI prediction results
    """

    def decorator(func):
        async def wrapper(self, symbol: str, *args, **kwargs):
            # Try to get from cache first
            provider = getattr(self, "provider_name", "unknown")
            cached = await redis_cache.get_ai_prediction(symbol, provider, kwargs)

            if cached:
                return cached

            # Call original function
            result = await func(self, symbol, *args, **kwargs)

            # Cache the result
            if result and not result.get("error"):
                await redis_cache.set_ai_prediction(symbol, provider, result, ttl, kwargs)

            return result

        return wrapper

    return decorator


# Export for use
__all__ = ["RedisCacheService", "redis_cache", "cache_ai_prediction"]
