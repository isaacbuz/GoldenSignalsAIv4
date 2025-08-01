"""
Tests for cache service.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import redis
import json
from datetime import datetime, timedelta
from src.services.cache_service import CacheService

@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    return Mock(spec=redis.Redis)

@pytest.fixture
def cache_service(mock_redis):
    """Create cache service with mock Redis."""
    return CacheService(redis_client=mock_redis)

class TestCacheService:
    """Test cache service functionality."""

    def test_set_cache(self, cache_service, mock_redis):
        """Test setting cache value."""
        key = "test_key"
        value = {"data": "test_value"}
        ttl = 300

        # Test successful cache set
        mock_redis.setex.return_value = True
        result = cache_service.set(key, value, ttl)

        assert result is True
        mock_redis.setex.assert_called_once()

        # Verify serialization
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == key
        assert call_args[0][1] == ttl
        assert json.loads(call_args[0][2]) == value

    def test_get_cache_hit(self, cache_service, mock_redis):
        """Test getting cached value (cache hit)."""
        key = "test_key"
        cached_value = {"data": "test_value"}

        # Mock Redis response
        mock_redis.get.return_value = json.dumps(cached_value).encode()

        result = cache_service.get(key)

        assert result == cached_value
        mock_redis.get.assert_called_once_with(key)

    def test_get_cache_miss(self, cache_service, mock_redis):
        """Test getting cached value (cache miss)."""
        key = "test_key"

        # Mock Redis response for cache miss
        mock_redis.get.return_value = None

        result = cache_service.get(key)

        assert result is None
        mock_redis.get.assert_called_once_with(key)

    def test_delete_cache(self, cache_service, mock_redis):
        """Test deleting cached value."""
        key = "test_key"

        # Mock successful deletion
        mock_redis.delete.return_value = 1

        result = cache_service.delete(key)

        assert result is True
        mock_redis.delete.assert_called_once_with(key)

    def test_exists_cache(self, cache_service, mock_redis):
        """Test checking if key exists in cache."""
        key = "test_key"

        # Test key exists
        mock_redis.exists.return_value = 1
        assert cache_service.exists(key) is True

        # Test key doesn't exist
        mock_redis.exists.return_value = 0
        assert cache_service.exists(key) is False

    def test_get_ttl(self, cache_service, mock_redis):
        """Test getting TTL of cached key."""
        key = "test_key"

        # Mock TTL response
        mock_redis.ttl.return_value = 120

        result = cache_service.get_ttl(key)

        assert result == 120
        mock_redis.ttl.assert_called_once_with(key)

    def test_cache_decorator(self, cache_service, mock_redis):
        """Test cache decorator functionality."""
        # Mock cache miss then hit
        mock_redis.get.side_effect = [None, json.dumps({"result": 42}).encode()]
        mock_redis.setex.return_value = True

        @cache_service.cache(ttl=60)
        def expensive_function(x):
            return {"result": x * 2}

        # First call - cache miss
        result1 = expensive_function(21)
        assert result1 == {"result": 42}
        assert mock_redis.get.call_count == 1
        assert mock_redis.setex.call_count == 1

        # Second call - cache hit
        result2 = expensive_function(21)
        assert result2 == {"result": 42}
        assert mock_redis.get.call_count == 2
        assert mock_redis.setex.call_count == 1  # No additional set

    def test_bulk_set(self, cache_service, mock_redis):
        """Test bulk setting multiple cache values."""
        items = {
            "key1": {"data": "value1"},
            "key2": {"data": "value2"},
            "key3": {"data": "value3"}
        }
        ttl = 300

        # Mock pipeline
        mock_pipeline = Mock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True, True, True]

        result = cache_service.bulk_set(items, ttl)

        assert result is True
        assert mock_pipeline.setex.call_count == 3
        mock_pipeline.execute.assert_called_once()

    def test_bulk_get(self, cache_service, mock_redis):
        """Test bulk getting multiple cache values."""
        keys = ["key1", "key2", "key3"]
        values = [
            json.dumps({"data": "value1"}).encode(),
            None,  # Cache miss
            json.dumps({"data": "value3"}).encode()
        ]

        mock_redis.mget.return_value = values

        result = cache_service.bulk_get(keys)

        assert result == {
            "key1": {"data": "value1"},
            "key2": None,
            "key3": {"data": "value3"}
        }
        mock_redis.mget.assert_called_once_with(keys)

    def test_clear_pattern(self, cache_service, mock_redis):
        """Test clearing cache by pattern."""
        pattern = "user:*"
        matching_keys = [b"user:1", b"user:2", b"user:3"]

        # Mock scan_iter
        mock_redis.scan_iter.return_value = matching_keys
        mock_redis.delete.return_value = 3

        result = cache_service.clear_pattern(pattern)

        assert result == 3
        mock_redis.scan_iter.assert_called_once_with(match=pattern)
        mock_redis.delete.assert_called_once_with(*matching_keys)

    def test_get_cache_stats(self, cache_service, mock_redis):
        """Test getting cache statistics."""
        # Mock Redis info
        mock_redis.info.return_value = {
            "used_memory_human": "10M",
            "connected_clients": 5,
            "total_commands_processed": 1000,
            "keyspace_hits": 800,
            "keyspace_misses": 200
        }

        stats = cache_service.get_stats()

        assert stats["memory_usage"] == "10M"
        assert stats["connected_clients"] == 5
        assert stats["hit_rate"] == 0.8  # 800 / (800 + 200)
        assert stats["total_commands"] == 1000

    def test_cache_error_handling(self, cache_service, mock_redis):
        """Test error handling in cache operations."""
        key = "test_key"

        # Simulate Redis connection error
        mock_redis.get.side_effect = redis.ConnectionError("Connection failed")

        # Should return None on error
        result = cache_service.get(key)
        assert result is None

        # Test set with error
        mock_redis.setex.side_effect = redis.ConnectionError("Connection failed")
        result = cache_service.set(key, {"data": "value"})
        assert result is False
