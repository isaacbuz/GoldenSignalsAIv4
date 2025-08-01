"""
Cache Manager for GoldenSignalsAI V2
Issue #197: Performance Optimization - Caching Strategies
"""

import asyncio
import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import redis.asyncio as redis
from redis.asyncio.lock import Lock
from redis.exceptions import RedisError

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different use cases"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    REFRESH_AHEAD = "refresh_ahead"


class CacheTier(Enum):
    """Multi-tier cache levels"""

    L1_MEMORY = "l1_memory"  # In-memory cache (fastest)
    L2_REDIS = "l2_redis"  # Redis cache (fast)
    L3_DATABASE = "l3_database"  # Database cache (persistent)


class CacheManager:
    """Advanced multi-tier cache manager with various strategies"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        l1_max_size: int = 10000,
        default_ttl: int = 3600,
        enable_metrics: bool = True,
    ):
        self.redis_url = redis_url
        self.l1_max_size = l1_max_size
        self.default_ttl = default_ttl
        self.enable_metrics = enable_metrics

        # L1 in-memory cache
        self.l1_cache: Dict[str, Tuple[Any, float, int]] = {}  # key: (value, timestamp, frequency)
        self.l1_access_order: List[str] = []

        # Redis connection pool
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None

        # Cache metrics
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "errors": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "cache_writes": 0,
        }

        # Background tasks
        self.refresh_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self):
        """Initialize cache connections"""
        try:
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url, max_connections=50, decode_responses=False
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            await self.redis_client.ping()
            logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            raise

    async def close(self):
        """Close cache connections"""
        # Cancel refresh tasks
        for task in self.refresh_tasks.values():
            task.cancel()

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        if self.redis_pool:
            await self.redis_pool.disconnect()

    def _generate_key(self, namespace: str, key: Union[str, dict]) -> str:
        """Generate cache key with namespace"""
        if isinstance(key, dict):
            # Sort dict keys for consistent hashing
            key_str = json.dumps(key, sort_keys=True)
        else:
            key_str = str(key)

        # Create hash for long keys
        if len(key_str) > 250:
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            return f"{namespace}:{key_hash}"

        return f"{namespace}:{key_str}"

    async def get(
        self, namespace: str, key: Union[str, dict], tier: CacheTier = CacheTier.L2_REDIS
    ) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_key(namespace, key)

        try:
            # Check L1 cache first
            if tier != CacheTier.L3_DATABASE:
                if cache_key in self.l1_cache:
                    value, timestamp, frequency = self.l1_cache[cache_key]
                    # Update frequency and access order
                    self.l1_cache[cache_key] = (value, timestamp, frequency + 1)
                    self.l1_access_order.remove(cache_key)
                    self.l1_access_order.append(cache_key)

                    self.metrics["hits"] += 1
                    self.metrics["l1_hits"] += 1
                    return value

            # Check L2 Redis cache
            if tier == CacheTier.L2_REDIS and self.redis_client:
                value_bytes = await self.redis_client.get(cache_key)
                if value_bytes:
                    value = pickle.loads(value_bytes)

                    # Populate L1 cache
                    await self._set_l1(cache_key, value)

                    self.metrics["hits"] += 1
                    self.metrics["l2_hits"] += 1
                    return value

            self.metrics["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.metrics["errors"] += 1
            return None

    async def set(
        self,
        namespace: str,
        key: Union[str, dict],
        value: Any,
        ttl: Optional[int] = None,
        strategy: CacheStrategy = CacheStrategy.TTL,
        tier: CacheTier = CacheTier.L2_REDIS,
    ):
        """Set value in cache with strategy"""
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl

        try:
            # Set in L1 cache
            if tier != CacheTier.L3_DATABASE:
                await self._set_l1(cache_key, value)

            # Set in L2 Redis cache
            if tier == CacheTier.L2_REDIS and self.redis_client:
                value_bytes = pickle.dumps(value)
                await self.redis_client.setex(cache_key, ttl, value_bytes)

                # Handle refresh-ahead strategy
                if strategy == CacheStrategy.REFRESH_AHEAD:
                    await self._setup_refresh_ahead(namespace, key, ttl)

            self.metrics["cache_writes"] += 1

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.metrics["errors"] += 1

    async def _set_l1(self, cache_key: str, value: Any):
        """Set value in L1 cache with LRU eviction"""
        # Check if we need to evict
        if len(self.l1_cache) >= self.l1_max_size:
            # Evict least recently used
            lru_key = self.l1_access_order.pop(0)
            del self.l1_cache[lru_key]
            self.metrics["evictions"] += 1

        # Add to cache
        self.l1_cache[cache_key] = (value, time.time(), 1)
        self.l1_access_order.append(cache_key)

    async def delete(self, namespace: str, key: Union[str, dict]):
        """Delete value from cache"""
        cache_key = self._generate_key(namespace, key)

        # Delete from L1
        if cache_key in self.l1_cache:
            del self.l1_cache[cache_key]
            self.l1_access_order.remove(cache_key)

        # Delete from L2
        if self.redis_client:
            await self.redis_client.delete(cache_key)

    async def delete_pattern(self, namespace: str, pattern: str):
        """Delete all keys matching pattern"""
        # Clear L1 cache for namespace
        keys_to_delete = [k for k in self.l1_cache.keys() if k.startswith(f"{namespace}:")]
        for key in keys_to_delete:
            del self.l1_cache[key]
            self.l1_access_order.remove(key)

        # Delete from Redis
        if self.redis_client:
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, match=f"{namespace}:{pattern}", count=100
                )
                if keys:
                    await self.redis_client.delete(*keys)
                if cursor == 0:
                    break

    async def _setup_refresh_ahead(self, namespace: str, key: Union[str, dict], ttl: int):
        """Setup refresh-ahead task for cache warming"""
        cache_key = self._generate_key(namespace, key)

        # Cancel existing task if any
        if cache_key in self.refresh_tasks:
            self.refresh_tasks[cache_key].cancel()

        async def refresh_task():
            try:
                # Wait until 80% of TTL has passed
                await asyncio.sleep(ttl * 0.8)

                # Trigger cache refresh callback if registered
                if hasattr(self, "_refresh_callbacks") and namespace in self._refresh_callbacks:
                    callback = self._refresh_callbacks[namespace]
                    new_value = await callback(key)
                    if new_value is not None:
                        await self.set(namespace, key, new_value, ttl, CacheStrategy.REFRESH_AHEAD)

            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Refresh-ahead error: {e}")

        self.refresh_tasks[cache_key] = asyncio.create_task(refresh_task())

    def cache_decorator(
        self,
        namespace: str,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None,
        strategy: CacheStrategy = CacheStrategy.TTL,
    ):
        """Decorator for caching function results"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    cache_key = {"args": args, "kwargs": kwargs}

                # Try to get from cache
                cached_value = await self.get(namespace, cache_key)
                if cached_value is not None:
                    return cached_value

                # Execute function
                result = await func(*args, **kwargs)

                # Cache the result
                await self.set(namespace, cache_key, result, ttl, strategy)

                return result

            return wrapper

        return decorator

    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        total_requests = self.metrics["hits"] + self.metrics["misses"]
        hit_rate = (self.metrics["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "l1_size": len(self.l1_cache),
            "l1_hit_rate": (self.metrics["l1_hits"] / self.metrics["hits"] * 100)
            if self.metrics["hits"] > 0
            else 0,
            **self.metrics,
        }

    async def warm_cache(
        self,
        namespace: str,
        keys: List[Union[str, dict]],
        values: List[Any],
        ttl: Optional[int] = None,
    ):
        """Pre-warm cache with multiple values"""
        tasks = []
        for key, value in zip(keys, values):
            task = self.set(namespace, key, value, ttl)
            tasks.append(task)

        await asyncio.gather(*tasks)
        logger.info(f"Warmed cache with {len(keys)} entries for namespace {namespace}")

    async def distributed_lock(self, lock_name: str, timeout: int = 10) -> Lock:
        """Get distributed lock using Redis"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")

        return self.redis_client.lock(f"lock:{lock_name}", timeout=timeout)


class CachedDataLoader:
    """Optimized data loader with intelligent caching"""

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    @staticmethod
    def _batch_key(symbols: List[str], start: datetime, end: datetime) -> str:
        """Generate key for batch data"""
        return f"{','.join(sorted(symbols))}:{start.isoformat()}:{end.isoformat()}"

    async def load_market_data(
        self, symbols: List[str], start: datetime, end: datetime, use_cache: bool = True
    ) -> Dict[str, Any]:
        """Load market data with intelligent caching"""
        if use_cache:
            # Check cache first
            cache_key = self._batch_key(symbols, start, end)
            cached_data = await self.cache.get("market_data", cache_key)
            if cached_data:
                return cached_data

        # Load data (simulate expensive operation)
        data = await self._fetch_market_data(symbols, start, end)

        if use_cache:
            # Cache with appropriate TTL based on data age
            if end.date() == datetime.now().date():
                ttl = 300  # 5 minutes for today's data
            else:
                ttl = 86400  # 24 hours for historical data

            await self.cache.set(
                "market_data",
                cache_key,
                data,
                ttl=ttl,
                strategy=CacheStrategy.REFRESH_AHEAD if ttl == 300 else CacheStrategy.TTL,
            )

        return data

    async def _fetch_market_data(
        self, symbols: List[str], start: datetime, end: datetime
    ) -> Dict[str, Any]:
        """Simulate fetching market data"""
        # This would be replaced with actual data fetching logic
        await asyncio.sleep(0.1)  # Simulate network delay
        return {
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "data": {},  # Actual data would go here
        }
