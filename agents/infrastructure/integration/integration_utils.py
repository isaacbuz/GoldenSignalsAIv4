"""
integration_utils.py

Provides integration adapters and external cache utilities for GoldenSignalsAI agents.
Includes:
- ExternalCache: Redis-backed cache for external model results
- ModelProviderAdapter and provider subclasses: Unified interface for Anthropic, Meta, Cohere, xAI/Grok, etc.

Migrated from /integration for unified access by agents and research modules.
"""
import asyncio
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import redis


class ExternalCache:
    """Simple Redis-backed cache for external model results (sentiment, embeddings, etc.)."""
    def __init__(self, redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")):
        self.client = redis.Redis.from_url(redis_url)
    def _make_key(self, prefix: str, data: Any) -> str:
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

class ModelProviderAdapter:
    """Abstract base class for all external model provider adapters."""
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError
    def generate_explanation(self, input: str) -> str:
        raise NotImplementedError
    def get_embeddings(self, text: str) -> List[float]:
        raise NotImplementedError
    def vision_analysis(self, image: Any) -> Any:
        raise NotImplementedError

class AnthropicClaudeAdapter(ModelProviderAdapter):
    """Adapter for Anthropic Claude models (text, sentiment, explanation)."""
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        return {"sentiment": "positive", "confidence": 0.95}
    def generate_explanation(self, input: str) -> str:
        return "This trade is recommended due to positive earnings and strong sentiment."

class MetaLlamaAdapter(ModelProviderAdapter):
    """Adapter for Meta Llama models (text, vision, multimodal)."""
    # Implement Meta Llama methods here
    pass

# Additional adapters (Cohere, xAI/Grok, etc.) can be added here as subclasses
