"""
external_cache.py
-----------------
Simple Redis-backed cache for external model results (sentiment, embeddings, etc.).
Used to speed up repeated inference and reduce API costs.
"""
import redis
import os
import hashlib
import json
from typing import Any, Optional

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class ExternalCache:
    def __init__(self, redis_url: str = REDIS_URL):
        self.client = redis.Redis.from_url(redis_url)

    def _make_key(self, prefix: str, data: Any) -> str:
        """Hash input data for a unique cache key."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get(self, prefix: str, data: Any) -> Optional[Any]:
        key = self._make_key(prefix, data)
        val = self.client.get(key)
        if val:
            try:
                return json.loads(val)
            except Exception:
                return val
        return None

    def set(self, prefix: str, data: Any, value: Any, ex: int = 3600):
        key = self._make_key(prefix, data)
        self.client.set(key, json.dumps(value), ex=ex)
