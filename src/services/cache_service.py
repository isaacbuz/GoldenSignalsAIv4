#!/usr/bin/env python3
"""
🚀 GoldenSignalsAI - Smart Caching Service
Multi-tier caching with intelligent TTL strategies

Features:
- Memory cache (L1)
- Redis cache (L2)
- Database cache (L3)
- Symbol-based TTL strategies
- Cache warming
- Cache analytics
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import redis
from cachetools import LRUCache, TTLCache

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache levels"""
    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"

class DataType(Enum):
    """Data types for TTL strategies"""
    QUOTE = "quote"
    HISTORICAL = "historical"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"
    INDICATOR = "indicator"
    SIGNAL = "signal"

@dataclass
class CacheConfig:
    """Cache configuration"""
    memory_size: int = 10000
    memory_ttl: int = 300  # 5 minutes
    redis_ttl: int = 600  # 10 minutes
    db_ttl: int = 3600  # 1 hour
    enable_warming: bool = True
    warming_interval: int = 300  # 5 minutes

class SmartCacheService:
    """Multi-tier caching service"""
    
    def __init__(self, config: CacheConfig = None):
        # L1: Memory cache
        self.memory_cache = {
            DataType.QUOTE: TTLCache(maxsize=5000, ttl=300),
            DataType.HISTORICAL: TTLCache(maxsize=1000, ttl=600),
            DataType.NEWS: TTLCache(maxsize=500, ttl=1800),
            DataType.FUNDAMENTAL: TTLCache(maxsize=1000, ttl=3600),
            DataType.INDICATOR: TTLCache(maxsize=2000, ttl=300),
            DataType.SIGNAL: TTLCache(maxsize=1000, ttl=60)
        }
        
        # L2: Redis cache
        self.redis_client = None
        
        # L3: Database cache (simulated)
        self.db_cache = {}
        
        # Cache statistics
        self.stats = {
            "hits": {"memory": 0, "redis": 0, "database": 0},
            "misses": 0,
            "writes": {"memory": 0, "redis": 0, "database": 0}
        }
        
        # Popular symbols for cache warming
        self.popular_symbols = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
            "META", "NVDA", "JPM", "V", "JNJ"
        ]
        
        # Background tasks will be started when needed
        self._background_tasks_started = False
        self.config = config or CacheConfig()
        
        # Initialize Redis
        self._init_redis()
    
    def _ensure_background_tasks(self):
        """Ensure background tasks are started when in async context"""
        if not self._background_tasks_started and self.config.enable_warming:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._cache_warming_task())
                self._background_tasks_started = True
            except RuntimeError:
                # No event loop running yet, tasks will start when one is available
                pass
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info("✅ Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            self.redis_client = None
    
    def _get_cache_key(self, data_type: DataType, key: str) -> str:
        """Generate cache key"""
        return f"{data_type.value}:{key}"
    
    def _get_ttl(self, data_type: DataType, symbol: str = None) -> Dict[str, int]:
        """Get TTL based on data type and symbol popularity"""
        base_ttl = {
            DataType.QUOTE: {"memory": 300, "redis": 600, "db": 3600},
            DataType.HISTORICAL: {"memory": 600, "redis": 1800, "db": 7200},
            DataType.NEWS: {"memory": 1800, "redis": 3600, "db": 86400},
            DataType.FUNDAMENTAL: {"memory": 3600, "redis": 7200, "db": 86400},
            DataType.INDICATOR: {"memory": 300, "redis": 600, "db": 3600},
            DataType.SIGNAL: {"memory": 60, "redis": 300, "db": 1800}
        }
        
        ttl = base_ttl.get(data_type, {"memory": 300, "redis": 600, "db": 3600})
        
        # Increase TTL for popular symbols
        if symbol and symbol in self.popular_symbols:
            ttl = {k: v * 2 for k, v in ttl.items()}
        
        return ttl
    
    async def get(
        self,
        data_type: DataType,
        key: str,
        fetch_func: Optional[Callable] = None
    ) -> Optional[Any]:
        """Get data from cache with automatic fallback"""
        # Ensure background tasks are running
        self._ensure_background_tasks()
        
        cache_key = self._get_cache_key(data_type, key)
        
        # L1: Check memory cache
        data = self._get_from_memory(data_type, cache_key)
        if data is not None:
            self.stats["hits"]["memory"] += 1
            return data
        
        # L2: Check Redis cache
        data = await self._get_from_redis(cache_key)
        if data is not None:
            self.stats["hits"]["redis"] += 1
            # Promote to memory cache
            self._set_to_memory(data_type, cache_key, data)
            return data
        
        # L3: Check database cache
        data = await self._get_from_database(cache_key)
        if data is not None:
            self.stats["hits"]["database"] += 1
            # Promote to higher caches
            await self._set_to_redis(cache_key, data, self._get_ttl(data_type, key)["redis"])
            self._set_to_memory(data_type, cache_key, data)
            return data
        
        # Cache miss
        self.stats["misses"] += 1
        
        # Fetch from source if function provided
        if fetch_func:
            try:
                data = await fetch_func()
                if data is not None:
                    await self.set(data_type, key, data)
                return data
            except Exception as e:
                logger.error(f"Fetch function error: {e}")
        
        return None
    
    async def set(
        self,
        data_type: DataType,
        key: str,
        data: Any,
        ttl_override: Optional[Dict[str, int]] = None
    ):
        """Set data in all cache levels"""
        cache_key = self._get_cache_key(data_type, key)
        ttl = ttl_override or self._get_ttl(data_type, key)
        
        # Write to all levels
        self._set_to_memory(data_type, cache_key, data)
        await self._set_to_redis(cache_key, data, ttl["redis"])
        await self._set_to_database(cache_key, data, ttl["db"])
    
    def _get_from_memory(self, data_type: DataType, key: str) -> Optional[Any]:
        """Get from memory cache"""
        cache = self.memory_cache.get(data_type)
        if cache and key in cache:
            return cache[key]
        return None
    
    def _set_to_memory(self, data_type: DataType, key: str, data: Any):
        """Set in memory cache"""
        cache = self.memory_cache.get(data_type)
        if cache:
            cache[key] = data
            self.stats["writes"]["memory"] += 1
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get from Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
        
        return None
    
    async def _set_to_redis(self, key: str, data: Any, ttl: int):
        """Set in Redis cache"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(data, default=str)
            )
            self.stats["writes"]["redis"] += 1
        except Exception as e:
            logger.debug(f"Redis set error: {e}")
    
    async def _get_from_database(self, key: str) -> Optional[Any]:
        """Get from database cache (simulated)"""
        # In production, this would query a real database
        return self.db_cache.get(key)
    
    async def _set_to_database(self, key: str, data: Any, ttl: int):
        """Set in database cache (simulated)"""
        # In production, this would write to a real database
        self.db_cache[key] = {
            "data": data,
            "expires": time.time() + ttl
        }
        self.stats["writes"]["database"] += 1
    
    async def invalidate(self, data_type: DataType, key: str):
        """Invalidate cache entry across all levels"""
        cache_key = self._get_cache_key(data_type, key)
        
        # Remove from memory
        cache = self.memory_cache.get(data_type)
        if cache and cache_key in cache:
            del cache[cache_key]
        
        # Remove from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(cache_key)
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")
        
        # Remove from database
        if cache_key in self.db_cache:
            del self.db_cache[cache_key]
    
    async def warm_cache(self, symbols: List[str] = None):
        """Warm cache with popular symbols"""
        symbols = symbols or self.popular_symbols
        
        logger.info(f"Warming cache for {len(symbols)} symbols")
        
        # Import here to avoid circular dependency
        from src.services.rate_limit_handler import get_rate_limit_handler
        handler = get_rate_limit_handler()
        
        # Warm quotes
        quotes = await handler.batch_get_quotes(symbols)
        for symbol, quote in quotes.items():
            if quote:
                await self.set(DataType.QUOTE, symbol, quote)
        
        logger.info(f"✅ Cache warmed with {len(quotes)} quotes")
    
    async def _cache_warming_task(self):
        """Background task for cache warming"""
        while True:
            try:
                await asyncio.sleep(self.config.warming_interval)
                await self.warm_cache()
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(self.stats["hits"].values())
        total_requests = total_hits + self.stats["misses"]
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "writes": self.stats["writes"],
            "hit_rate": f"{hit_rate:.1f}%",
            "total_requests": total_requests,
            "memory_size": {
                dt.value: len(cache) for dt, cache in self.memory_cache.items()
            }
        }
    
    async def clear_expired(self):
        """Clear expired entries from database cache"""
        now = time.time()
        expired_keys = []
        
        for key, entry in self.db_cache.items():
            if entry["expires"] < now:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.db_cache[key]
        
        logger.info(f"Cleared {len(expired_keys)} expired entries")

# Singleton instance
_cache_service: Optional[SmartCacheService] = None

def get_cache_service() -> SmartCacheService:
    """Get or create cache service singleton"""
    global _cache_service
    if _cache_service is None:
        _cache_service = SmartCacheService()
    return _cache_service 