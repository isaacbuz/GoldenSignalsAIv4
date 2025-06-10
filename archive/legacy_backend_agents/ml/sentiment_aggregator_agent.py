"""
SentimentAggregatorAgent (Upgraded)
Aggregates sentiment from pluggable SentimentSource classes (Twitter, Reddit, News, etc.) asynchronously and caches results.
"""
import asyncio
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from archive.legacy_backend_agents.sentiment_source import TwitterSentimentSource, RedditSentimentSource
try:
    import redis
except ImportError:
    redis = None

class SentimentAggregatorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.cache = self._init_cache()
        self.sources = [TwitterSentimentSource(), RedditSentimentSource()]  # Add more as needed

    def _init_cache(self):
        if redis is not None:
            try:
                r = redis.Redis(host='localhost', port=6379, db=0)
                r.ping()
                return r
            except Exception:
                pass
        return dict()

    async def fetch_all(self, symbol: str) -> List[Dict[str, Any]]:
        tasks = [self._fetch_source(source, symbol) for source in self.sources]
        return await asyncio.gather(*tasks)

    async def _fetch_source(self, source, symbol: str) -> Dict[str, Any]:
        # For real async APIs, make async HTTP calls here
        return source.fetch(symbol)

    async def aggregate(self, symbol: str) -> Dict[str, Any]:
        cache_key = f"sentiment:{symbol}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        sentiments = await self.fetch_all(symbol)
        avg_sentiment = sum(s.get('sentiment', 0) for s in sentiments) / len(sentiments)
        result = {
            "symbol": symbol,
            "sources": sentiments,
            "average_sentiment": avg_sentiment,
            "explanation": ", ".join(f"{s['platform']}: {s['sentiment']:.2f}" for s in sentiments)
        }
        if isinstance(self.cache, dict):
            self.cache[cache_key] = result
        else:
            self.cache.set(cache_key, str(result))
        return result

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = data.get('symbol', 'AAPL')
        loop = asyncio.get_event_loop()
        sentiment = loop.run_until_complete(self.aggregate(symbol))
        return sentiment

    def train(self, historical_data: Dict[str, Any]) -> None:
        """Placeholder for retraining or calibration logic."""
        pass
