"""
Redis Cache Service for Agent Results
Optimizes signal generation by caching agent analysis results
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
import hashlib
import redis
from redis.exceptions import RedisError
import os

logger = logging.getLogger(__name__)


class RedisCacheService:
    """Redis-based caching service for agent results and market data"""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis cache service
        
        Args:
            redis_url: Redis connection URL (defaults to env var or localhost)
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self._client: Optional[redis.Redis] = None
        self._connect()
        
    def _connect(self) -> None:
        """Establish Redis connection with retry logic"""
        try:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self._client.ping()
            logger.info("✅ Redis cache connected successfully")
        except RedisError as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Cache will be disabled.")
            self._client = None
    
    def _is_connected(self) -> bool:
        """Check if Redis is connected and available"""
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except RedisError:
            return False
    
    def _generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """
        Generate cache key from prefix and parameters
        
        Args:
            prefix: Key prefix (e.g., 'agent_result', 'market_data')
            params: Parameters to include in key
            
        Returns:
            Cache key string
        """
        # Sort params for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:8]
        return f"{prefix}:{param_hash}"
    
    def get_agent_result(
        self,
        agent_name: str,
        symbol: str,
        timeframe: str,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached agent result
        
        Args:
            agent_name: Name of the agent
            symbol: Trading symbol
            timeframe: Analysis timeframe
            additional_params: Additional parameters
            
        Returns:
            Cached result or None if not found/expired
        """
        if not self._is_connected():
            return None
            
        params = {
            'agent': agent_name,
            'symbol': symbol,
            'timeframe': timeframe,
            **(additional_params or {})
        }
        
        key = self._generate_key('agent_result', params)
        
        try:
            result = self._client.get(key)
            if result:
                data = json.loads(result)
                logger.debug(f"Cache HIT for {key}")
                return data
            logger.debug(f"Cache MISS for {key}")
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set_agent_result(
        self,
        agent_name: str,
        symbol: str,
        timeframe: str,
        result: Dict[str, Any],
        ttl_seconds: int = 300,  # 5 minutes default
        additional_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache agent result
        
        Args:
            agent_name: Name of the agent
            symbol: Trading symbol
            timeframe: Analysis timeframe
            result: Result to cache
            ttl_seconds: Time to live in seconds
            additional_params: Additional parameters
            
        Returns:
            Success status
        """
        if not self._is_connected():
            return False
            
        params = {
            'agent': agent_name,
            'symbol': symbol,
            'timeframe': timeframe,
            **(additional_params or {})
        }
        
        key = self._generate_key('agent_result', params)
        
        # Add metadata
        cache_data = {
            'result': result,
            'cached_at': datetime.now().isoformat(),
            'ttl': ttl_seconds
        }
        
        try:
            self._client.setex(
                key,
                ttl_seconds,
                json.dumps(cache_data)
            )
            logger.debug(f"Cached result for {key} (TTL: {ttl_seconds}s)")
            return True
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get_market_data(
        self,
        symbol: str,
        interval: str,
        period: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached market data"""
        if not self._is_connected():
            return None
            
        params = {
            'symbol': symbol,
            'interval': interval,
            'period': period
        }
        
        key = self._generate_key('market_data', params)
        
        try:
            result = self._client.get(key)
            if result:
                return json.loads(result)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Market data cache error: {e}")
            return None
    
    def set_market_data(
        self,
        symbol: str,
        interval: str,
        period: str,
        data: Dict[str, Any],
        ttl_seconds: int = 60  # 1 minute for market data
    ) -> bool:
        """Cache market data"""
        if not self._is_connected():
            return False
            
        params = {
            'symbol': symbol,
            'interval': interval,
            'period': period
        }
        
        key = self._generate_key('market_data', params)
        
        try:
            self._client.setex(
                key,
                ttl_seconds,
                json.dumps(data)
            )
            return True
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Market data cache error: {e}")
            return False
    
    def invalidate_agent_cache(self, agent_name: str) -> int:
        """
        Invalidate all cache entries for a specific agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Number of keys deleted
        """
        if not self._is_connected():
            return 0
            
        try:
            pattern = f"agent_result:*"
            deleted = 0
            
            # Scan for matching keys
            for key in self._client.scan_iter(match=pattern):
                # Check if this key belongs to the agent
                data = self._client.get(key)
                if data:
                    try:
                        cache_data = json.loads(data)
                        if cache_data.get('result', {}).get('agent_name') == agent_name:
                            self._client.delete(key)
                            deleted += 1
                    except json.JSONDecodeError:
                        pass
                        
            logger.info(f"Invalidated {deleted} cache entries for agent {agent_name}")
            return deleted
        except RedisError as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._is_connected():
            return {'connected': False}
            
        try:
            info = self._client.info()
            stats = {
                'connected': True,
                'used_memory': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': 0.0
            }
            
            # Calculate hit rate
            total_ops = stats['keyspace_hits'] + stats['keyspace_misses']
            if total_ops > 0:
                stats['hit_rate'] = (stats['keyspace_hits'] / total_ops) * 100
                
            return stats
        except RedisError as e:
            logger.error(f"Stats error: {e}")
            return {'connected': False, 'error': str(e)}
    
    def clear_all_cache(self) -> bool:
        """Clear all cache entries (use with caution)"""
        if not self._is_connected():
            return False
            
        try:
            self._client.flushdb()
            logger.warning("⚠️ All cache entries cleared")
            return True
        except RedisError as e:
            logger.error(f"Cache clear error: {e}")
            return False


# Global cache instance
_cache_instance: Optional[RedisCacheService] = None


def get_cache_service() -> RedisCacheService:
    """Get or create cache service singleton"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCacheService()
    return _cache_instance


# Decorator for caching agent results
def cache_agent_result(ttl_seconds: int = 300):
    """
    Decorator to cache agent results
    
    Args:
        ttl_seconds: Cache TTL in seconds
    """
    def decorator(func):
        def wrapper(self, symbol: str, *args, **kwargs):
            cache = get_cache_service()
            
            # Try to get from cache
            agent_name = self.__class__.__name__
            timeframe = kwargs.get('timeframe', '1d')
            
            cached = cache.get_agent_result(
                agent_name=agent_name,
                symbol=symbol,
                timeframe=timeframe
            )
            
            if cached:
                logger.info(f"Using cached result for {agent_name}:{symbol}")
                return cached['result']
            
            # Execute function
            result = func(self, symbol, *args, **kwargs)
            
            # Cache the result
            cache.set_agent_result(
                agent_name=agent_name,
                symbol=symbol,
                timeframe=timeframe,
                result=result,
                ttl_seconds=ttl_seconds
            )
            
            return result
        return wrapper
    return decorator