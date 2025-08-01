"""
Real-time News and Sentiment Integration
Integrates news feeds and analyzes sentiment for trading signals
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import feedparser
import yfinance as yf
from textblob import TextBlob

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article"""

    article_id: str
    title: str
    content: str
    source: str
    url: str
    published: datetime
    symbols: List[str]
    sentiment: float  # -1 to 1
    relevance: float  # 0 to 1
    category: str


class NewsIntegration:
    """Integrates real-time news and sentiment analysis"""

    def __init__(self):
        self.news_sources = {
            "reuters": "https://www.reuters.com/finance/rss",
            "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
            "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories",
        }
        self.sentiment_cache = {}
        self.article_cache = {}

    async def fetch_latest_news(self, symbols: List[str]) -> List[NewsArticle]:
        """Fetch latest news for given symbols"""
        all_articles = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for source_name, feed_url in self.news_sources.items():
                task = self._fetch_feed(session, source_name, feed_url, symbols)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for articles in results:
                if isinstance(articles, list):
                    all_articles.extend(articles)

        # Deduplicate and sort by published date
        unique_articles = self._deduplicate_articles(all_articles)
        unique_articles.sort(key=lambda x: x.published, reverse=True)

        return unique_articles[:50]  # Return top 50 most recent

    async def _fetch_feed(
        self, session: aiohttp.ClientSession, source: str, feed_url: str, symbols: List[str]
    ) -> List[NewsArticle]:
        """Fetch and parse RSS feed"""
        try:
            async with session.get(feed_url, timeout=10) as response:
                content = await response.text()

            feed = feedparser.parse(content)
            articles = []

            for entry in feed.entries[:20]:  # Process top 20 entries
                article = await self._process_entry(entry, source, symbols)
                if article:
                    articles.append(article)

            return articles

        except Exception as e:
            logger.error(f"Error fetching {source} feed: {e}")
            return []

    async def _process_entry(
        self, entry: Any, source: str, symbols: List[str]
    ) -> Optional[NewsArticle]:
        """Process feed entry into NewsArticle"""
        try:
            # Extract basic info
            title = entry.get("title", "")
            content = entry.get("summary", "")
            url = entry.get("link", "")

            # Parse published date
            published = datetime.now()
            if hasattr(entry, "published_parsed"):
                published = datetime.fromtimestamp(entry.published_parsed)

            # Check relevance to symbols
            relevant_symbols = self._extract_symbols(title + " " + content, symbols)
            if not relevant_symbols:
                return None

            # Analyze sentiment
            sentiment = self._analyze_sentiment(title + " " + content)

            # Calculate relevance
            relevance = len(relevant_symbols) / len(symbols) if symbols else 0.5

            article = NewsArticle(
                article_id=f"{source}_{hash(url)}",
                title=title,
                content=content[:500],  # Limit content length
                source=source,
                url=url,
                published=published,
                symbols=relevant_symbols,
                sentiment=sentiment,
                relevance=relevance,
                category=self._categorize_article(title, content),
            )

            return article

        except Exception as e:
            logger.error(f"Error processing entry: {e}")
            return None

    def _extract_symbols(self, text: str, symbols: List[str]) -> List[str]:
        """Extract mentioned symbols from text"""
        text_upper = text.upper()
        mentioned = []

        for symbol in symbols:
            if symbol.upper() in text_upper:
                mentioned.append(symbol)

        # Also check for company names
        # In production, use a more sophisticated NER system

        return mentioned

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        # Cache sentiment to avoid recomputation
        text_hash = hash(text)
        if text_hash in self.sentiment_cache:
            return self.sentiment_cache[text_hash]

        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity  # -1 to 1
            self.sentiment_cache[text_hash] = sentiment
            return sentiment
        except:
            return 0.0

    def _categorize_article(self, title: str, content: str) -> str:
        """Categorize article based on content"""
        text = (title + " " + content).lower()

        categories = {
            "earnings": ["earnings", "revenue", "profit", "quarterly"],
            "merger": ["merger", "acquisition", "takeover", "buyout"],
            "regulation": ["regulation", "sec", "regulatory", "compliance"],
            "product": ["launch", "product", "release", "announce"],
            "market": ["market", "trading", "stock", "shares"],
            "economic": ["fed", "inflation", "gdp", "economic"],
        }

        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category

        return "general"

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles"""
        seen_titles = set()
        unique = []

        for article in articles:
            title_hash = hash(article.title.lower())
            if title_hash not in seen_titles:
                seen_titles.add(title_hash)
                unique.append(article)

        return unique

    async def get_sentiment_summary(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get sentiment summary for symbol"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Fetch recent articles
        articles = await self.fetch_latest_news([symbol])
        recent_articles = [a for a in articles if a.published > cutoff_time]

        if not recent_articles:
            return {
                "symbol": symbol,
                "sentiment": 0.0,
                "article_count": 0,
                "positive_ratio": 0.0,
                "categories": {},
            }

        # Calculate aggregate sentiment
        sentiments = [a.sentiment for a in recent_articles]
        avg_sentiment = sum(sentiments) / len(sentiments)

        # Count positive/negative
        positive = sum(1 for s in sentiments if s > 0.1)
        negative = sum(1 for s in sentiments if s < -0.1)

        # Category breakdown
        category_counts = {}
        for article in recent_articles:
            category_counts[article.category] = category_counts.get(article.category, 0) + 1

        return {
            "symbol": symbol,
            "sentiment": avg_sentiment,
            "article_count": len(recent_articles),
            "positive_ratio": positive / len(recent_articles) if recent_articles else 0,
            "negative_ratio": negative / len(recent_articles) if recent_articles else 0,
            "categories": category_counts,
            "trend": self._calculate_sentiment_trend(recent_articles),
        }

    def _calculate_sentiment_trend(self, articles: List[NewsArticle]) -> str:
        """Calculate sentiment trend"""
        if len(articles) < 2:
            return "stable"

        # Sort by time
        sorted_articles = sorted(articles, key=lambda x: x.published)

        # Compare first half vs second half
        mid = len(sorted_articles) // 2
        first_half = sum(a.sentiment for a in sorted_articles[:mid]) / mid
        second_half = sum(a.sentiment for a in sorted_articles[mid:]) / (len(sorted_articles) - mid)

        diff = second_half - first_half

        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
