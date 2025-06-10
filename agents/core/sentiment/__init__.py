from typing import Dict, Any
"""
__init__.py
Purpose: Marks the sentiment agents directory as a Python subpackage. No runtime logic is present in this file.
"""

"""
Sentiment analysis agents for market analysis.
"""

from .news.finbert_sentiment_agent import FinBERTSentimentAgent
from .social.social_sentiment_agent import SocialSentimentAgent
from .meta.sentiment_aggregator import SentimentAggregatorAgent

__all__ = [
    'FinBERTSentimentAgent',
    'SocialSentimentAgent',
    'SentimentAggregatorAgent'
]
