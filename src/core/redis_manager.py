"""
Redis Cache Manager for GoldenSignalsAI V3

Handles real-time data caching, WebSocket state, and high-frequency operations.
"""

import asyncio
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable

import redis.asyncio as redis
from loguru import logger

from .config import settings


class RedisManager:
    """
    Redis manager for caching, real-time data, and WebSocket state management
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self._initialized = False
        
        # Cache key prefixes for organization
        self.PREFIXES = {
            "market_data": "market:",
            "agent_cache": "agent:",
            "signal_cache": "signal:",
            "websocket_state": "ws:",
            "session": "session:",
            "rate_limit": "rate:",
            "temp_data": "temp:"
        }

    # Compatibility alias so other components can access the raw Redis client
    @property
    def redis(self):
        """Return underlying redis client (alias for redis_client)."""
        return self.redis_client
    
    async def initialize(self) -> None:
        """Initialize Redis connection and pubsub"""
        try:
            # Parse Redis URL
            redis_url = settings.redis.url
            
            # Create Redis client with connection pooling
            self.redis_client = redis.from_url(
                redis_url,
                max_connections=settings.redis.max_connections,
                socket_timeout=settings.redis.socket_timeout,
                decode_responses=False,  # We'll handle encoding manually
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Initialize pubsub for real-time messaging
            self.pubsub = self.redis_client.pubsub()
            
            self._initialized = True
            logger.info("Redis manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check Redis connectivity"""
        try:
            if not self.redis_client:
                return False
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False
    
    def _make_key(self, prefix: str, key: str) -> str:
        """Create a prefixed Redis key"""
        return f"{self.PREFIXES[prefix]}{key}"
    
    async def _serialize(self, data: Any) -> bytes:
        """Serialize data for Redis storage"""
        if isinstance(data, (dict, list)):
            return json.dumps(data, default=str).encode('utf-8')
        elif isinstance(data, str):
            return data.encode('utf-8')
        else:
            return pickle.dumps(data)
    
    async def _deserialize(self, data: bytes, as_json: bool = True) -> Any:
        """Deserialize data from Redis"""
        if data is None:
            return None
        
        try:
            if as_json:
                return json.loads(data.decode('utf-8'))
            else:
                return pickle.loads(data)
        except (json.JSONDecodeError, pickle.UnpicklingError):
            # Fallback to string
            return data.decode('utf-8')
    
    # Market Data Caching
    async def cache_market_data(
        self, 
        symbol: str, 
        data: Dict[str, Any], 
        ttl: int = 300  # 5 minutes
    ) -> None:
        """Cache market data with TTL"""
        key = self._make_key("market_data", f"{symbol}:latest")
        serialized_data = await self._serialize(data)
        await self.redis_client.setex(key, ttl, serialized_data)
    
    async def get_cached_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data"""
        key = self._make_key("market_data", f"{symbol}:latest")
        data = await self.redis_client.get(key)
        return await self._deserialize(data) if data else None
    
    async def cache_ohlcv_data(
        self, 
        symbol: str, 
        timeframe: str, 
        ohlcv_list: List[Dict[str, Any]],
        ttl: int = 60
    ) -> None:
        """Cache OHLCV data for specific timeframe"""
        key = self._make_key("market_data", f"{symbol}:ohlcv:{timeframe}")
        serialized_data = await self._serialize(ohlcv_list)
        await self.redis_client.setex(key, ttl, serialized_data)
    
    async def get_cached_ohlcv_data(
        self, 
        symbol: str, 
        timeframe: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached OHLCV data"""
        key = self._make_key("market_data", f"{symbol}:ohlcv:{timeframe}")
        data = await self.redis_client.get(key)
        return await self._deserialize(data) if data else None
    
    # Agent State Caching
    async def cache_agent_state(
        self, 
        agent_id: str, 
        state_data: Dict[str, Any],
        ttl: int = 3600  # 1 hour
    ) -> None:
        """Cache agent state temporarily"""
        key = self._make_key("agent_cache", f"{agent_id}:state")
        serialized_data = await self._serialize(state_data)
        await self.redis_client.setex(key, ttl, serialized_data)
    
    async def get_cached_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get cached agent state"""
        key = self._make_key("agent_cache", f"{agent_id}:state")
        data = await self.redis_client.get(key)
        return await self._deserialize(data) if data else None
    
    async def cache_agent_performance(
        self, 
        agent_id: str, 
        performance_data: Dict[str, Any],
        ttl: int = 300  # 5 minutes
    ) -> None:
        """Cache agent performance metrics"""
        key = self._make_key("agent_cache", f"{agent_id}:performance")
        serialized_data = await self._serialize(performance_data)
        await self.redis_client.setex(key, ttl, serialized_data)
    
    async def get_cached_agent_performance(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get cached agent performance"""
        key = self._make_key("agent_cache", f"{agent_id}:performance")
        data = await self.redis_client.get(key)
        return await self._deserialize(data) if data else None
    
    # Signal Caching
    async def cache_latest_signals(
        self, 
        symbol: str, 
        signals: List[Dict[str, Any]],
        ttl: int = 60
    ) -> None:
        """Cache latest signals for a symbol"""
        key = self._make_key("signal_cache", f"{symbol}:latest")
        serialized_data = await self._serialize(signals)
        await self.redis_client.setex(key, ttl, serialized_data)
    
    async def get_cached_latest_signals(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached latest signals"""
        key = self._make_key("signal_cache", f"{symbol}:latest")
        data = await self.redis_client.get(key)
        return await self._deserialize(data) if data else None
    
    async def add_signal_to_stream(self, symbol: str, signal_data: Dict[str, Any]) -> None:
        """Add signal to real-time stream"""
        key = self._make_key("signal_cache", f"{symbol}:stream")
        serialized_data = await self._serialize(signal_data)
        
        # Add to stream and keep only last 100 signals
        pipe = self.redis_client.pipeline()
        pipe.lpush(key, serialized_data)
        pipe.ltrim(key, 0, 99)  # Keep only 100 most recent
        pipe.expire(key, 3600)  # 1 hour TTL
        await pipe.execute()
    
    async def get_signal_stream(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get signals from stream"""
        key = self._make_key("signal_cache", f"{symbol}:stream")
        data_list = await self.redis_client.lrange(key, 0, limit - 1)
        
        signals = []
        for data in data_list:
            signal = await self._deserialize(data)
            if signal:
                signals.append(signal)
        
        return signals
    
    # WebSocket State Management
    async def register_websocket_client(
        self, 
        client_id: str, 
        symbol: str, 
        client_info: Dict[str, Any]
    ) -> None:
        """Register WebSocket client for symbol"""
        key = self._make_key("websocket_state", f"{symbol}:clients")
        serialized_info = await self._serialize(client_info)
        await self.redis_client.hset(key, client_id, serialized_info)
        await self.redis_client.expire(key, 7200)  # 2 hours TTL
    
    async def unregister_websocket_client(self, client_id: str, symbol: str) -> None:
        """Unregister WebSocket client"""
        key = self._make_key("websocket_state", f"{symbol}:clients")
        await self.redis_client.hdel(key, client_id)
    
    async def get_websocket_clients(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Get all WebSocket clients for symbol"""
        key = self._make_key("websocket_state", f"{symbol}:clients")
        clients_data = await self.redis_client.hgetall(key)
        
        clients = {}
        for client_id, data in clients_data.items():
            client_info = await self._deserialize(data)
            clients[client_id.decode('utf-8')] = client_info
        
        return clients
    
    # Real-time Pub/Sub Messaging
    async def publish_signal(self, symbol: str, signal_data: Dict[str, Any]) -> None:
        """Publish signal to subscribers"""
        channel = f"signals:{symbol}"
        serialized_data = await self._serialize(signal_data)
        await self.redis_client.publish(channel, serialized_data)
    
    async def publish_market_data(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """Publish market data update"""
        channel = f"market_data:{symbol}"
        serialized_data = await self._serialize(market_data)
        await self.redis_client.publish(channel, serialized_data)
    
    async def subscribe_to_signals(
        self, 
        symbols: List[str], 
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Subscribe to signal updates"""
        channels = [f"signals:{symbol}" for symbol in symbols]
        await self.pubsub.subscribe(*channels)
        
        async for message in self.pubsub.listen():
            if message['type'] == 'message':
                channel = message['channel'].decode('utf-8')
                symbol = channel.split(':')[1]
                data = await self._deserialize(message['data'])
                await callback(symbol, data)
    
    # Rate Limiting
    async def check_rate_limit(
        self, 
        identifier: str, 
        max_requests: int, 
        window_seconds: int
    ) -> bool:
        """Check if request is within rate limit"""
        key = self._make_key("rate_limit", identifier)
        current_time = int(datetime.utcnow().timestamp())
        window_start = current_time - window_seconds
        
        # Remove old entries and count current requests
        pipe = self.redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.expire(key, window_seconds)
        results = await pipe.execute()
        
        current_count = results[1]
        
        if current_count >= max_requests:
            return False
        
        # Add current request
        await self.redis_client.zadd(key, {str(current_time): current_time})
        return True
    
    # Session Management
    async def store_session(
        self, 
        session_id: str, 
        session_data: Dict[str, Any],
        ttl: int = 3600
    ) -> None:
        """Store session data"""
        key = self._make_key("session", session_id)
        serialized_data = await self._serialize(session_data)
        await self.redis_client.setex(key, ttl, serialized_data)
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        key = self._make_key("session", session_id)
        data = await self.redis_client.get(key)
        return await self._deserialize(data) if data else None
    
    async def delete_session(self, session_id: str) -> None:
        """Delete session"""
        key = self._make_key("session", session_id)
        await self.redis_client.delete(key)
    
    # Temporary Data Storage
    async def store_temp_data(
        self, 
        key: str, 
        data: Any, 
        ttl: int = 300
    ) -> None:
        """Store temporary data with TTL"""
        redis_key = self._make_key("temp_data", key)
        serialized_data = await self._serialize(data)
        await self.redis_client.setex(redis_key, ttl, serialized_data)
    
    async def get_temp_data(self, key: str) -> Optional[Any]:
        """Get temporary data"""
        redis_key = self._make_key("temp_data", key)
        data = await self.redis_client.get(redis_key)
        return await self._deserialize(data, as_json=False) if data else None
    
    # Utility Methods
    async def increment_counter(
        self, 
        key: str, 
        amount: int = 1, 
        ttl: Optional[int] = None
    ) -> int:
        """Increment a counter"""
        redis_key = self._make_key("temp_data", f"counter:{key}")
        value = await self.redis_client.incrby(redis_key, amount)
        
        if ttl:
            await self.redis_client.expire(redis_key, ttl)
        
        return value
    
    async def get_counter(self, key: str) -> int:
        """Get counter value"""
        redis_key = self._make_key("temp_data", f"counter:{key}")
        value = await self.redis_client.get(redis_key)
        return int(value) if value else 0
    
    async def set_flag(self, key: str, ttl: int = 3600) -> None:
        """Set a temporary flag"""
        redis_key = self._make_key("temp_data", f"flag:{key}")
        await self.redis_client.setex(redis_key, ttl, "1")
    
    async def check_flag(self, key: str) -> bool:
        """Check if flag is set"""
        redis_key = self._make_key("temp_data", f"flag:{key}")
        return await self.redis_client.exists(redis_key) > 0
    
    async def clear_flag(self, key: str) -> None:
        """Clear a flag"""
        redis_key = self._make_key("temp_data", f"flag:{key}")
        await self.redis_client.delete(redis_key)
    
    # Cleanup and Maintenance
    async def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data and return statistics"""
        cleanup_stats = {
            "expired_keys": 0,
            "memory_freed": 0
        }
        
        # Get memory usage before cleanup
        memory_before = await self.redis_client.memory_usage("dummy_key") or 0
        
        # Clean up expired WebSocket client registrations
        ws_pattern = self._make_key("websocket_state", "*")
        ws_keys = await self.redis_client.keys(ws_pattern)
        
        for key in ws_keys:
            ttl = await self.redis_client.ttl(key)
            if ttl == -1:  # No TTL set
                await self.redis_client.expire(key, 7200)  # Set 2 hour default
        
        # Get memory usage after cleanup
        memory_after = await self.redis_client.memory_usage("dummy_key") or 0
        cleanup_stats["memory_freed"] = max(0, memory_before - memory_after)
        
        logger.info(f"Redis cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        info = await self.redis_client.info()
        
        stats = {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory_human", "0B"),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "cache_hit_rate": 0.0
        }
        
        # Calculate cache hit rate
        hits = stats["keyspace_hits"]
        misses = stats["keyspace_misses"]
        if hits + misses > 0:
            stats["cache_hit_rate"] = hits / (hits + misses)
        
        return stats
    
    async def close(self) -> None:
        """Close Redis connections"""
        try:
            if self.pubsub:
                await self.pubsub.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self._initialized = False
            logger.info("Redis connections closed")
            
        except Exception as e:
            logger.error(f"Error closing Redis connections: {str(e)}") 