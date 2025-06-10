"""
SentimentSource Interface
Defines a pluggable interface for fetching sentiment from any external API or data source.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

class SentimentSource(ABC):
    @abstractmethod
    def fetch(self, symbol: str) -> Dict[str, Any]:
        """Fetch sentiment data for the given symbol."""
        pass

# Example implementation for Twitter (placeholder)
class TwitterSentimentSource(SentimentSource):
    def fetch(self, symbol: str) -> Dict[str, Any]:
        # TODO: Integrate Twitter API
        return {"platform": "twitter", "symbol": symbol, "sentiment": 0.1, "sample": "This is a bullish tweet."}

# Example implementation for Reddit (placeholder)
class RedditSentimentSource(SentimentSource):
    def fetch(self, symbol: str) -> Dict[str, Any]:
        # TODO: Integrate Reddit API
        return {"platform": "reddit", "symbol": symbol, "sentiment": -0.05, "sample": "This is a bearish post."}
