"""
News & Event Impact RAG
Links news events to historical price movements for predictive insights
Issue #181: RAG-2: Implement News & Event Impact RAG
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NewsCategory(Enum):
    """Categories of news events"""
    EARNINGS = "earnings"
    FED = "federal_reserve"
    ECONOMIC = "economic_data"
    GEOPOLITICAL = "geopolitical"
    CORPORATE = "corporate_action"
    SECTOR = "sector_news"
    ANALYST = "analyst_rating"
    REGULATORY = "regulatory"


class SentimentLevel(Enum):
    """Sentiment levels for news"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class NewsEvent:
    """Represents a news event with impact data"""
    id: str
    timestamp: datetime
    headline: str
    content: str
    category: NewsCategory
    sentiment: SentimentLevel
    symbols: List[str]
    source: str
    # Historical impact data
    price_impact_1h: float  # Price change 1 hour after
    price_impact_1d: float  # Price change 1 day after
    price_impact_1w: float  # Price change 1 week after
    volume_spike: float  # Volume increase ratio
    volatility_change: float  # IV change
    market_reaction_time: int  # Minutes to peak reaction

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'headline': self.headline,
            'category': self.category.value,
            'sentiment': self.sentiment.value,
            'symbols': self.symbols,
            'source': self.source,
            'price_impact_1h': self.price_impact_1h,
            'price_impact_1d': self.price_impact_1d,
            'price_impact_1w': self.price_impact_1w,
            'volume_spike': self.volume_spike,
            'volatility_change': self.volatility_change,
            'market_reaction_time': self.market_reaction_time
        }

    def to_search_text(self) -> str:
        """Convert to searchable text for embedding"""
        return f"""
        {self.headline}
        Category: {self.category.value}
        Sentiment: {self.sentiment.name}
        Symbols: {', '.join(self.symbols)}
        Impact: {self.price_impact_1d:.2f}% in 24h
        Volume: {self.volume_spike:.1f}x normal
        Reaction time: {self.market_reaction_time} minutes
        """


class NewsImpactRAG:
    """
    RAG system for linking news to historical price movements
    Provides predictive insights for news-driven trading
    """

    def __init__(self, use_mock_db: bool = True):
        """Initialize the News Impact RAG system"""
        self.use_mock_db = use_mock_db
        self.news_events: List[NewsEvent] = []
        self.embeddings: Dict[str, np.ndarray] = {}
        self.sentiment_patterns: Dict[str, List[float]] = {}

        if use_mock_db:
            self._load_mock_news_events()

    def _load_mock_news_events(self):
        """Load mock historical news events with impact data"""
        mock_events = [
            # Fed Rate Decision
            NewsEvent(
                id="fed_2022_75bp",
                timestamp=datetime(2022, 6, 15, 14, 0),
                headline="Fed Raises Rates by 75 Basis Points, Largest Hike Since 1994",
                content="Federal Reserve raises interest rates by 0.75 percentage points...",
                category=NewsCategory.FED,
                sentiment=SentimentLevel.NEGATIVE,
                symbols=["SPY", "QQQ", "TLT"],
                source="Reuters",
                price_impact_1h=-2.1,
                price_impact_1d=-3.5,
                price_impact_1w=-5.2,
                volume_spike=2.8,
                volatility_change=0.35,
                market_reaction_time=15
            ),
            # Positive Earnings Surprise
            NewsEvent(
                id="aapl_earnings_beat_2023",
                timestamp=datetime(2023, 2, 2, 16, 5),
                headline="Apple Reports Record Q1 Earnings, Beats on Revenue and EPS",
                content="Apple Inc. reported quarterly earnings that exceeded analyst expectations...",
                category=NewsCategory.EARNINGS,
                sentiment=SentimentLevel.VERY_POSITIVE,
                symbols=["AAPL"],
                source="Bloomberg",
                price_impact_1h=3.2,
                price_impact_1d=5.1,
                price_impact_1w=4.8,
                volume_spike=3.5,
                volatility_change=-0.15,
                market_reaction_time=5
            ),
            # Geopolitical Event
            NewsEvent(
                id="ukraine_invasion_2022",
                timestamp=datetime(2022, 2, 24, 4, 0),
                headline="Russia Launches Military Operation in Ukraine",
                content="Global markets plunge as Russia begins military action...",
                category=NewsCategory.GEOPOLITICAL,
                sentiment=SentimentLevel.VERY_NEGATIVE,
                symbols=["SPY", "VIX", "GLD", "OIL"],
                source="AP",
                price_impact_1h=-2.8,
                price_impact_1d=-4.2,
                price_impact_1w=-1.5,
                volume_spike=4.2,
                volatility_change=0.85,
                market_reaction_time=30
            ),
            # Economic Data Miss
            NewsEvent(
                id="cpi_hot_2022",
                timestamp=datetime(2022, 9, 13, 8, 30),
                headline="US Inflation Comes in Hotter Than Expected at 8.3%",
                content="Consumer Price Index rises more than forecast, dashing hopes...",
                category=NewsCategory.ECONOMIC,
                sentiment=SentimentLevel.VERY_NEGATIVE,
                symbols=["SPY", "DXY", "TLT"],
                source="BLS",
                price_impact_1h=-2.5,
                price_impact_1d=-4.3,
                price_impact_1w=-6.1,
                volume_spike=2.2,
                volatility_change=0.42,
                market_reaction_time=2
            ),
            # Analyst Upgrade
            NewsEvent(
                id="nvda_upgrade_2023",
                timestamp=datetime(2023, 5, 24, 7, 0),
                headline="Goldman Sachs Upgrades NVIDIA to Buy, Cites AI Boom",
                content="Goldman Sachs raises NVIDIA price target to $500...",
                category=NewsCategory.ANALYST,
                sentiment=SentimentLevel.POSITIVE,
                symbols=["NVDA"],
                source="Goldman Sachs",
                price_impact_1h=1.8,
                price_impact_1d=3.2,
                price_impact_1w=7.5,
                volume_spike=1.8,
                volatility_change=0.05,
                market_reaction_time=10
            ),
            # Regulatory News
            NewsEvent(
                id="crypto_sec_2023",
                timestamp=datetime(2023, 6, 5, 9, 0),
                headline="SEC Sues Binance and Coinbase for Securities Violations",
                content="Securities and Exchange Commission takes action against major crypto exchanges...",
                category=NewsCategory.REGULATORY,
                sentiment=SentimentLevel.VERY_NEGATIVE,
                symbols=["COIN", "BTC", "ETH"],
                source="SEC",
                price_impact_1h=-8.5,
                price_impact_1d=-12.3,
                price_impact_1w=-15.2,
                volume_spike=5.2,
                volatility_change=0.95,
                market_reaction_time=5
            )
        ]

        self.news_events.extend(mock_events)
        self._generate_mock_embeddings()

    def _generate_mock_embeddings(self):
        """Generate embeddings for news events"""
        for event in self.news_events:
            # Create a simple embedding based on news features
            embedding = np.array([
                event.sentiment.value / 2.0,  # Normalized sentiment
                1.0 if event.category == NewsCategory.FED else 0.0,
                1.0 if event.category == NewsCategory.EARNINGS else 0.0,
                1.0 if event.category == NewsCategory.GEOPOLITICAL else 0.0,
                abs(event.price_impact_1d) / 10.0,  # Normalized impact
                event.volume_spike / 5.0,  # Normalized volume
                event.volatility_change,  # Volatility change
                event.market_reaction_time / 60.0,  # Normalized reaction time
            ])
            self.embeddings[event.id] = embedding

    def _calculate_similarity(self, query_embedding: np.ndarray,
                            news_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(query_embedding, news_embedding)
        norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(news_embedding)
        if norm_product == 0:
            return 0.0
        return dot_product / norm_product

    async def analyze_sentiment(self, text: str) -> SentimentLevel:
        """Analyze sentiment of news text"""
        # Simple keyword-based sentiment analysis
        # In production, use transformer models like FinBERT

        text_lower = text.lower()

        very_negative_words = ['crash', 'plunge', 'collapse', 'crisis', 'bankruptcy']
        negative_words = ['fall', 'decline', 'miss', 'cut', 'weak', 'concern']
        positive_words = ['rise', 'gain', 'beat', 'strong', 'upgrade', 'growth']
        very_positive_words = ['surge', 'soar', 'record', 'breakthrough', 'exceptional']

        score = 0
        for word in very_negative_words:
            if word in text_lower:
                score -= 2
        for word in negative_words:
            if word in text_lower:
                score -= 1
        for word in positive_words:
            if word in text_lower:
                score += 1
        for word in very_positive_words:
            if word in text_lower:
                score += 2

        if score <= -2:
            return SentimentLevel.VERY_NEGATIVE
        elif score < 0:
            return SentimentLevel.NEGATIVE
        elif score == 0:
            return SentimentLevel.NEUTRAL
        elif score <= 2:
            return SentimentLevel.POSITIVE
        else:
            return SentimentLevel.VERY_POSITIVE

    def _categorize_news(self, headline: str, content: str) -> NewsCategory:
        """Categorize news based on content"""
        text = (headline + " " + content).lower()

        if any(word in text for word in ['fed', 'fomc', 'powell', 'monetary policy']):
            return NewsCategory.FED
        elif any(word in text for word in ['earnings', 'revenue', 'eps', 'quarterly']):
            return NewsCategory.EARNINGS
        elif any(word in text for word in ['cpi', 'gdp', 'unemployment', 'inflation']):
            return NewsCategory.ECONOMIC
        elif any(word in text for word in ['war', 'conflict', 'sanctions', 'geopolitical']):
            return NewsCategory.GEOPOLITICAL
        elif any(word in text for word in ['upgrade', 'downgrade', 'analyst', 'rating']):
            return NewsCategory.ANALYST
        elif any(word in text for word in ['sec', 'regulatory', 'investigation', 'lawsuit']):
            return NewsCategory.REGULATORY
        elif any(word in text for word in ['merger', 'acquisition', 'buyout', 'deal']):
            return NewsCategory.CORPORATE
        else:
            return NewsCategory.SECTOR

    async def get_similar_news_impacts(self, news_item: Dict[str, Any],
                                     top_k: int = 5) -> Dict[str, Any]:
        """Get historical impacts of similar news events"""

        # Analyze the new news item
        sentiment = await self.analyze_sentiment(news_item.get('text', news_item.get('headline', '')))
        category = self._categorize_news(
            news_item.get('headline', ''),
            news_item.get('text', news_item.get('content', ''))
        )

        # Create embedding for the query
        query_embedding = np.array([
            sentiment.value / 2.0,
            1.0 if category == NewsCategory.FED else 0.0,
            1.0 if category == NewsCategory.EARNINGS else 0.0,
            1.0 if category == NewsCategory.GEOPOLITICAL else 0.0,
            0.5,  # Unknown impact
            1.0,  # Normal volume assumption
            0.0,  # Unknown volatility change
            0.5,  # Average reaction time
        ])

        # Find similar historical news
        similarities = []
        for event in self.news_events:
            # Filter by symbol if provided
            if 'symbols' in news_item:
                if not any(s in event.symbols for s in news_item['symbols']):
                    continue

            news_embedding = self.embeddings[event.id]
            similarity = self._calculate_similarity(query_embedding, news_embedding)
            similarities.append((similarity, event))

        # Sort by similarity and get top K
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_matches = similarities[:top_k]

        # Calculate expected impact
        if top_matches:
            avg_price_move = np.mean([e.price_impact_1d for _, e in top_matches])
            avg_reaction_time = np.mean([e.market_reaction_time for _, e in top_matches])
            avg_volume_spike = np.mean([e.volume_spike for _, e in top_matches])
            confidence = np.mean([s for s, _ in top_matches])
        else:
            avg_price_move = 0.0
            avg_reaction_time = 30
            avg_volume_spike = 1.0
            confidence = 0.0

        return {
            'query': news_item,
            'sentiment': sentiment.name,
            'category': category.value,
            'similar_events': [
                {
                    'similarity': float(sim),
                    'headline': event.headline,
                    'date': event.timestamp.isoformat(),
                    'price_impact_1h': event.price_impact_1h,
                    'price_impact_1d': event.price_impact_1d,
                    'price_impact_1w': event.price_impact_1w,
                    'reaction_time': event.market_reaction_time
                }
                for sim, event in top_matches
            ],
            'expected_impact': {
                'price_move_1d': float(avg_price_move),
                'reaction_time_minutes': int(avg_reaction_time),
                'volume_spike': float(avg_volume_spike),
                'confidence': float(confidence)
            },
            'trading_recommendation': self._generate_trading_recommendation(
                sentiment, category, avg_price_move, confidence
            )
        }

    def _generate_trading_recommendation(self, sentiment: SentimentLevel,
                                       category: NewsCategory,
                                       expected_move: float,
                                       confidence: float) -> Dict[str, Any]:
        """Generate trading recommendation based on news analysis"""

        # Determine action
        if confidence < 0.5:
            action = "hold"
            size = 0
        elif expected_move > 2.0 and sentiment.value > 0:
            action = "buy"
            size = min(1.0, confidence)
        elif expected_move < -2.0 and sentiment.value < 0:
            action = "sell"
            size = min(1.0, confidence)
        elif abs(expected_move) > 3.0:
            action = "hedge"
            size = 0.5
        else:
            action = "monitor"
            size = 0

        # Determine urgency
        if category in [NewsCategory.FED, NewsCategory.ECONOMIC]:
            urgency = "immediate"  # These move markets fast
        elif category == NewsCategory.EARNINGS:
            urgency = "high"
        else:
            urgency = "medium"

        # Risk management
        stop_loss = abs(expected_move) * 0.5
        take_profit = abs(expected_move) * 1.5

        return {
            'action': action,
            'size': float(size),
            'urgency': urgency,
            'expected_move': float(expected_move),
            'confidence': float(confidence),
            'stop_loss_percent': float(stop_loss),
            'take_profit_percent': float(take_profit),
            'hold_period': "1-3 days",
            'rationale': f"{sentiment.name} {category.value} news typically moves {expected_move:.1f}%"
        }

    async def get_category_statistics(self, category: NewsCategory) -> Dict[str, Any]:
        """Get historical statistics for a news category"""
        category_events = [e for e in self.news_events if e.category == category]

        if not category_events:
            return {
                'category': category.value,
                'event_count': 0,
                'avg_impact': 0.0,
                'avg_reaction_time': 0
            }

        impacts_1d = [e.price_impact_1d for e in category_events]
        reaction_times = [e.market_reaction_time for e in category_events]
        volume_spikes = [e.volume_spike for e in category_events]

        return {
            'category': category.value,
            'event_count': len(category_events),
            'avg_impact_1d': float(np.mean(impacts_1d)),
            'impact_std': float(np.std(impacts_1d)),
            'max_impact': float(np.max(np.abs(impacts_1d))),
            'avg_reaction_time': float(np.mean(reaction_times)),
            'avg_volume_spike': float(np.mean(volume_spikes)),
            'typical_sentiment': self._get_typical_sentiment(category_events)
        }

    def _get_typical_sentiment(self, events: List[NewsEvent]) -> str:
        """Get the most common sentiment for a set of events"""
        if not events:
            return "NEUTRAL"

        sentiments = [e.sentiment.value for e in events]
        avg_sentiment = np.mean(sentiments)

        if avg_sentiment <= -1.5:
            return "VERY_NEGATIVE"
        elif avg_sentiment <= -0.5:
            return "NEGATIVE"
        elif avg_sentiment <= 0.5:
            return "NEUTRAL"
        elif avg_sentiment <= 1.5:
            return "POSITIVE"
        else:
            return "VERY_POSITIVE"

    async def track_news_accuracy(self, news_id: str, actual_impact: Dict[str, float]):
        """Track prediction accuracy for continuous learning"""
        # In production, this would update the model with actual outcomes
        logger.info(f"Tracking accuracy for news {news_id}: {actual_impact}")
        # Store for future model updates


# Demo function
async def demo_news_impact_rag():
    """Demonstrate the News Impact RAG system"""
    rag = NewsImpactRAG(use_mock_db=True)

    print("News Impact RAG Demo")
    print("="*60)

    # Test different news scenarios
    test_news = [
        {
            "headline": "Federal Reserve Signals Aggressive Rate Hikes Ahead",
            "text": "Fed Chair Powell indicates multiple 50bp hikes likely",
            "symbols": ["SPY", "QQQ"],
            "timestamp": datetime.now()
        },
        {
            "headline": "Apple Beats Earnings Expectations, Raises Guidance",
            "text": "Tech giant reports record iPhone sales, services growth",
            "symbols": ["AAPL"],
            "timestamp": datetime.now()
        },
        {
            "headline": "Geopolitical Tensions Rise as Conflict Escalates",
            "text": "Markets tumble on fears of wider conflict",
            "symbols": ["SPY", "GLD", "OIL"],
            "timestamp": datetime.now()
        }
    ]

    for news in test_news:
        print(f"\n{'='*60}")
        print(f"News: {news['headline']}")
        print(f"Symbols: {', '.join(news['symbols'])}")

        result = await rag.get_similar_news_impacts(news, top_k=3)

        print(f"\nSentiment: {result['sentiment']}")
        print(f"Category: {result['category']}")
        print(f"\nExpected Impact:")
        print(f"  Price Move (1d): {result['expected_impact']['price_move_1d']:.2f}%")
        print(f"  Reaction Time: {result['expected_impact']['reaction_time_minutes']} minutes")
        print(f"  Volume Spike: {result['expected_impact']['volume_spike']:.1f}x")
        print(f"  Confidence: {result['expected_impact']['confidence']:.1%}")

        print(f"\nTrading Recommendation:")
        rec = result['trading_recommendation']
        print(f"  Action: {rec['action'].upper()}")
        print(f"  Size: {rec['size']:.1%} of portfolio")
        print(f"  Urgency: {rec['urgency']}")
        print(f"  Stop Loss: {rec['stop_loss_percent']:.1f}%")
        print(f"  Take Profit: {rec['take_profit_percent']:.1f}%")

        if result['similar_events']:
            print(f"\nSimilar Historical Events:")
            for i, event in enumerate(result['similar_events'][:2], 1):
                print(f"  {i}. {event['headline'][:60]}...")
                print(f"     Impact: {event['price_impact_1d']:.2f}% (Similarity: {event['similarity']:.1%})")

    # Show category statistics
    print(f"\n{'='*60}")
    print("Category Statistics:")
    for category in [NewsCategory.FED, NewsCategory.EARNINGS, NewsCategory.GEOPOLITICAL]:
        stats = await rag.get_category_statistics(category)
        if stats['event_count'] > 0:
            print(f"\n{category.value.upper()}:")
            print(f"  Events: {stats['event_count']}")
            print(f"  Avg Impact: {stats['avg_impact_1d']:.2f}% ± {stats['impact_std']:.2f}%")
            print(f"  Avg Reaction Time: {stats['avg_reaction_time']:.0f} minutes")


if __name__ == "__main__":
    asyncio.run(demo_news_impact_rag())
