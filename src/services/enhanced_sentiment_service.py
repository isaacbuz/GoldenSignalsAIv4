"""
Enhanced Sentiment Analysis Service
Integrates X/Twitter API, News API, Reddit API, and FinBERT for comprehensive sentiment analysis
"""

import asyncio
import aiohttp
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import json
import hashlib
from enum import Enum

# Import utilities
from src.utils.timezone_utils import now_utc
from src.utils.error_recovery import with_retry, error_recovery, ErrorSeverity

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    TWITTER = "twitter"
    NEWS = "news"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"


@dataclass
class SentimentData:
    """Container for sentiment data from various sources"""
    source: SentimentSource
    symbol: str
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    volume: int  # Number of mentions/articles
    timestamp: datetime
    raw_data: Dict[str, Any]
    keywords: List[str]


class EnhancedSentimentService:
    """Enhanced sentiment analysis service with multiple data sources"""
    
    def __init__(self):
        # API Keys from environment
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN', '')
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID', '')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        
        # Cache settings
        self.cache = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Sentiment keywords
        self.bullish_keywords = [
            'bullish', 'buy', 'long', 'calls', 'moon', 'rocket', 'pump',
            'breakout', 'squeeze', 'gamma', 'diamond hands', 'hodl',
            'to the moon', 'tendies', 'yolo', 'strong', 'upgrade'
        ]
        
        self.bearish_keywords = [
            'bearish', 'sell', 'short', 'puts', 'crash', 'dump', 'tank',
            'breakdown', 'resistance', 'overbought', 'paper hands',
            'bubble', 'overvalued', 'weak', 'downgrade'
        ]
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.reddit_token: Optional[str] = None
        self.reddit_token_expiry: Optional[datetime] = None
    
    async def initialize(self):
        """Initialize the service"""
        self.session = aiohttp.ClientSession()
        logger.info("Enhanced Sentiment Service initialized")
    
    async def close(self):
        """Close the service"""
        if self.session:
            await self.session.close()
    
    async def get_aggregated_sentiment(
        self, 
        symbol: str,
        include_sources: List[SentimentSource] = None
    ) -> Dict[str, Any]:
        """Get aggregated sentiment from all sources"""
        
        # Check cache
        cache_key = f"{symbol}_{now_utc().strftime('%Y%m%d%H%M')}"
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if now_utc() - cache_time < self.cache_ttl:
                return cached_data
        
        # Default to all sources
        if not include_sources:
            include_sources = list(SentimentSource)
        
        # Gather sentiment from all sources concurrently
        tasks = []
        
        if SentimentSource.TWITTER in include_sources and self.twitter_bearer_token:
            tasks.append(self._get_twitter_sentiment(symbol))
        
        if SentimentSource.NEWS in include_sources and self.news_api_key:
            tasks.append(self._get_news_sentiment(symbol))
        
        if SentimentSource.REDDIT in include_sources and self.reddit_client_id:
            tasks.append(self._get_reddit_sentiment(symbol))
        
        if SentimentSource.STOCKTWITS in include_sources:
            tasks.append(self._get_stocktwits_sentiment(symbol))
        
        # Collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_sentiments = []
        for result in results:
            if isinstance(result, SentimentData):
                valid_sentiments.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error collecting sentiment: {result}")
        
        # Aggregate sentiment
        aggregated = self._aggregate_sentiments(valid_sentiments)
        
        # Add symbol and timestamp
        aggregated['symbol'] = symbol
        aggregated['timestamp'] = now_utc().isoformat()
        
        # Cache result
        self.cache[cache_key] = (aggregated, now_utc())
        
        return aggregated
    
    async def _get_twitter_sentiment(self, symbol: str) -> SentimentData:
        """Get sentiment from X/Twitter API v2"""
        
        if not self.twitter_bearer_token:
            raise ValueError("Twitter Bearer Token not configured")
        
        # Twitter API v2 endpoint
        url = "https://api.twitter.com/2/tweets/search/recent"
        
        # Search query - cashtag and company mentions
        query = f"${symbol} OR #{symbol} -is:retweet lang:en"
        
        headers = {
            "Authorization": f"Bearer {self.twitter_bearer_token}",
            "User-Agent": "GoldenSignalsAI/1.0"
        }
        
        params = {
            "query": query,
            "max_results": 100,
            "tweet.fields": "created_at,public_metrics,author_id",
            "start_time": (now_utc() - timedelta(hours=24)).isoformat()
        }
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                tweets = data.get('data', [])
                
                # Analyze sentiment
                sentiment_scores = []
                total_engagement = 0
                keywords_found = []
                
                for tweet in tweets:
                    text = tweet.get('text', '').lower()
                    metrics = tweet.get('public_metrics', {})
                    
                    # Simple sentiment scoring
                    bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text)
                    bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text)
                    
                    # Weight by engagement
                    engagement = (
                        metrics.get('retweet_count', 0) * 2 +
                        metrics.get('like_count', 0) +
                        metrics.get('reply_count', 0) +
                        metrics.get('quote_count', 0)
                    )
                    
                    if bullish_count > bearish_count:
                        sentiment_scores.append((1.0, engagement))
                        keywords_found.extend([k for k in self.bullish_keywords if k in text])
                    elif bearish_count > bullish_count:
                        sentiment_scores.append((-1.0, engagement))
                        keywords_found.extend([k for k in self.bearish_keywords if k in text])
                    else:
                        sentiment_scores.append((0.0, engagement))
                    
                    total_engagement += engagement
                
                # Calculate weighted sentiment
                if sentiment_scores and total_engagement > 0:
                    weighted_sentiment = sum(
                        score * engagement for score, engagement in sentiment_scores
                    ) / total_engagement
                else:
                    weighted_sentiment = 0.0
                
                return SentimentData(
                    source=SentimentSource.TWITTER,
                    symbol=symbol,
                    score=np.clip(weighted_sentiment, -1, 1),
                    confidence=min(len(tweets) / 100, 1.0),  # More tweets = higher confidence
                    volume=len(tweets),
                    timestamp=now_utc(),
                    raw_data={"tweet_count": len(tweets), "engagement": total_engagement},
                    keywords=list(set(keywords_found))[:10]
                )
            else:
                logger.error(f"Twitter API error: {response.status}")
                raise Exception(f"Twitter API error: {response.status}")
    
    def _aggregate_sentiments(self, sentiments: List[SentimentData]) -> Dict[str, Any]:
        """Aggregate sentiment data from multiple sources"""
        
        if not sentiments:
            return {
                'overall_score': 0.0,
                'overall_confidence': 0.0,
                'overall_label': 'Neutral',
                'sources': {},
                'keywords': [],
                'volume': {'total': 0}
            }
        
        # Weight by confidence and volume
        total_weight = 0
        weighted_score = 0
        source_data = {}
        all_keywords = []
        total_volume = 0
        
        for sentiment in sentiments:
            # Weight = confidence * log(volume + 1)
            weight = sentiment.confidence * np.log(sentiment.volume + 1)
            weighted_score += sentiment.score * weight
            total_weight += weight
            
            # Store source-specific data
            source_data[sentiment.source.value] = {
                'score': round(sentiment.score, 3),
                'confidence': round(sentiment.confidence, 3),
                'volume': sentiment.volume,
                'keywords': sentiment.keywords[:5]
            }
            
            all_keywords.extend(sentiment.keywords)
            total_volume += sentiment.volume
        
        # Calculate overall sentiment
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        overall_confidence = min(total_weight / 10, 1.0)  # Normalize confidence
        
        # Determine label
        if overall_score >= 0.3:
            label = 'Very Bullish' if overall_score >= 0.6 else 'Bullish'
        elif overall_score <= -0.3:
            label = 'Very Bearish' if overall_score <= -0.6 else 'Bearish'
        else:
            label = 'Neutral'
        
        # Get top keywords
        keyword_counts = defaultdict(int)
        for keyword in all_keywords:
            keyword_counts[keyword] += 1
        
        top_keywords = sorted(
            keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'overall_score': round(overall_score, 3),
            'overall_confidence': round(overall_confidence, 3),
            'overall_label': label,
            'sources': source_data,
            'keywords': [k for k, _ in top_keywords],
            'volume': {
                'total': total_volume,
                'by_source': {s.source.value: s.volume for s in sentiments}
            }
        }
    
    @error_recovery.with_recovery(
        fallback=lambda self, symbol: SentimentData(
            source=SentimentSource.NEWS,
            symbol=symbol,
            score=0.0,
            confidence=0.0,
            volume=0,
            timestamp=now_utc(),
            raw_data={},
            keywords=[]
        ),
        severity=ErrorSeverity.MEDIUM
    )
    async def _get_news_sentiment(self, symbol: str) -> SentimentData:
        """Get sentiment from News API"""
        
        if not self.news_api_key:
            raise ValueError("News API key not configured")
        
        # Get company name for better search
        company_names = {
            'AAPL': 'Apple',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'META': 'Meta Facebook',
            'AMZN': 'Amazon',
            'SPY': 'S&P 500'
        }
        
        company = company_names.get(symbol, symbol)
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'"{company}" OR "{symbol}" stock market',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 50,
            'from': (now_utc() - timedelta(days=1)).strftime('%Y-%m-%d')
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                articles = data.get('articles', [])
                
                # Analyze headlines and descriptions
                sentiment_scores = []
                keywords_found = []
                
                for article in articles:
                    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                    
                    # Enhanced news-specific keywords
                    positive_news = ['upgrade', 'beat', 'exceed', 'surge', 'rally', 'gain', 'profit', 'breakthrough']
                    negative_news = ['downgrade', 'miss', 'fall', 'plunge', 'loss', 'concern', 'investigation', 'lawsuit']
                    
                    positive_count = sum(1 for word in positive_news if word in text)
                    negative_count = sum(1 for word in negative_news if word in text)
                    
                    if positive_count > negative_count:
                        sentiment_scores.append(0.8)
                        keywords_found.extend([w for w in positive_news if w in text])
                    elif negative_count > positive_count:
                        sentiment_scores.append(-0.8)
                        keywords_found.extend([w for w in negative_news if w in text])
                    else:
                        sentiment_scores.append(0.0)
                
                # Calculate average sentiment
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
                
                return SentimentData(
                    source=SentimentSource.NEWS,
                    symbol=symbol,
                    score=np.clip(avg_sentiment, -1, 1),
                    confidence=min(len(articles) / 20, 1.0),
                    volume=len(articles),
                    timestamp=now_utc(),
                    raw_data={"article_count": len(articles)},
                    keywords=list(set(keywords_found))[:10]
                )
            else:
                logger.error(f"News API error: {response.status}")
                raise Exception(f"News API error: {response.status}")
    
    async def _get_reddit_auth_token(self) -> str:
        """Get Reddit OAuth token"""
        if self.reddit_token and self.reddit_token_expiry and now_utc() < self.reddit_token_expiry:
            return self.reddit_token
        
        # Get new token
        auth = aiohttp.BasicAuth(self.reddit_client_id, self.reddit_client_secret)
        data = {
            'grant_type': 'client_credentials',
            'scope': 'read'
        }
        headers = {'User-Agent': 'GoldenSignalsAI/1.0'}
        
        async with self.session.post(
            'https://www.reddit.com/api/v1/access_token',
            auth=auth,
            data=data,
            headers=headers
        ) as response:
            if response.status == 200:
                token_data = await response.json()
                self.reddit_token = token_data['access_token']
                self.reddit_token_expiry = now_utc() + timedelta(seconds=token_data['expires_in'] - 60)
                return self.reddit_token
            else:
                raise Exception(f"Reddit auth error: {response.status}")
    
    @error_recovery.with_recovery(
        fallback=lambda self, symbol: SentimentData(
            source=SentimentSource.REDDIT,
            symbol=symbol,
            score=0.0,
            confidence=0.0,
            volume=0,
            timestamp=now_utc(),
            raw_data={},
            keywords=[]
        ),
        severity=ErrorSeverity.MEDIUM
    )
    async def _get_reddit_sentiment(self, symbol: str) -> SentimentData:
        """Get sentiment from Reddit (WSB and other finance subreddits)"""
        
        if not self.reddit_client_id:
            raise ValueError("Reddit credentials not configured")
        
        # Get auth token
        token = await self._get_reddit_auth_token()
        
        headers = {
            'Authorization': f'Bearer {token}',
            'User-Agent': 'GoldenSignalsAI/1.0'
        }
        
        # Search multiple subreddits
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
        all_posts = []
        
        for subreddit in subreddits:
            url = f'https://oauth.reddit.com/r/{subreddit}/search'
            params = {
                'q': symbol,
                'sort': 'hot',
                'limit': 25,
                't': 'day',
                'restrict_sr': 'true'
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = data.get('data', {}).get('children', [])
                    all_posts.extend(posts)
        
        # Analyze sentiment
        sentiment_scores = []
        total_score = 0
        keywords_found = []
        
        for post in all_posts:
            post_data = post.get('data', {})
            title = post_data.get('title', '').lower()
            selftext = post_data.get('selftext', '').lower()
            text = f"{title} {selftext}"
            score = post_data.get('score', 0)
            num_comments = post_data.get('num_comments', 0)
            
            # Reddit-specific sentiment
            bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text)
            bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text)
            
            # Weight by engagement (upvotes + comments)
            engagement = score + num_comments * 2
            
            if bullish_count > bearish_count:
                sentiment_scores.append((0.8, engagement))
                keywords_found.extend([k for k in self.bullish_keywords if k in text])
            elif bearish_count > bullish_count:
                sentiment_scores.append((-0.8, engagement))
                keywords_found.extend([k for k in self.bearish_keywords if k in text])
            else:
                sentiment_scores.append((0.0, engagement))
            
            total_score += engagement
        
        # Calculate weighted sentiment
        if sentiment_scores and total_score > 0:
            weighted_sentiment = sum(
                score * engagement for score, engagement in sentiment_scores
            ) / total_score
        else:
            weighted_sentiment = 0.0
        
        return SentimentData(
            source=SentimentSource.REDDIT,
            symbol=symbol,
            score=np.clip(weighted_sentiment, -1, 1),
            confidence=min(len(all_posts) / 50, 1.0),
            volume=len(all_posts),
            timestamp=now_utc(),
            raw_data={"post_count": len(all_posts), "total_score": total_score},
            keywords=list(set(keywords_found))[:10]
        )
    
    @error_recovery.with_recovery(
        fallback=lambda self, symbol: SentimentData(
            source=SentimentSource.STOCKTWITS,
            symbol=symbol,
            score=0.0,
            confidence=0.0,
            volume=0,
            timestamp=now_utc(),
            raw_data={},
            keywords=[]
        ),
        severity=ErrorSeverity.LOW
    )
    async def _get_stocktwits_sentiment(self, symbol: str) -> SentimentData:
        """Get sentiment from StockTwits (no API key required)"""
        
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                messages = data.get('messages', [])
                
                # StockTwits provides sentiment labels
                bullish_count = 0
                bearish_count = 0
                total_likes = 0
                
                for message in messages:
                    sentiment = message.get('entities', {}).get('sentiment', {})
                    basic_sentiment = sentiment.get('basic', '')
                    likes = message.get('likes', {}).get('total', 0)
                    
                    if basic_sentiment == 'Bullish':
                        bullish_count += 1
                    elif basic_sentiment == 'Bearish':
                        bearish_count += 1
                    
                    total_likes += likes
                
                # Calculate sentiment ratio
                total_sentiment = bullish_count + bearish_count
                if total_sentiment > 0:
                    sentiment_score = (bullish_count - bearish_count) / total_sentiment
                else:
                    sentiment_score = 0.0
                
                return SentimentData(
                    source=SentimentSource.STOCKTWITS,
                    symbol=symbol,
                    score=np.clip(sentiment_score, -1, 1),
                    confidence=min(len(messages) / 30, 1.0),
                    volume=len(messages),
                    timestamp=now_utc(),
                    raw_data={
                        "bullish": bullish_count,
                        "bearish": bearish_count,
                        "total_likes": total_likes
                    },
                    keywords=[]
                )
            else:
                logger.error(f"StockTwits API error: {response.status}")
                raise Exception(f"StockTwits API error: {response.status}")
    
    async def get_market_sentiment_heatmap(
        self,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Get sentiment heatmap for multiple symbols"""
        
        tasks = [self.get_aggregated_sentiment(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        heatmap_data = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, dict):
                heatmap_data.append({
                    'symbol': symbol,
                    'sentiment': result.get('overall_score', 0),
                    'confidence': result.get('overall_confidence', 0),
                    'label': result.get('overall_label', 'Unknown'),
                    'volume': result.get('volume', {}).get('total', 0)
                })
            else:
                logger.error(f"Error getting sentiment for {symbol}: {result}")
                heatmap_data.append({
                    'symbol': symbol,
                    'sentiment': 0,
                    'confidence': 0,
                    'label': 'Error',
                    'volume': 0
                })
        
        # Sort by sentiment score
        heatmap_data.sort(key=lambda x: x['sentiment'], reverse=True)
        
        return {
            'heatmap': heatmap_data,
            'timestamp': now_utc().isoformat(),
            'summary': {
                'most_bullish': heatmap_data[0] if heatmap_data else None,
                'most_bearish': heatmap_data[-1] if heatmap_data else None,
                'average_sentiment': np.mean([d['sentiment'] for d in heatmap_data]) if heatmap_data else 0
            }
        }


# Singleton instance
    
    @error_recovery.with_recovery(
        fallback=lambda self, symbol: SentimentData(
            source=SentimentSource.NEWS,
            symbol=symbol,
            score=0.0,
            confidence=0.0,
            volume=0,
            timestamp=now_utc(),
            raw_data={},
            keywords=[]
        ),
        severity=ErrorSeverity.MEDIUM
    )
    async def _get_news_sentiment(self, symbol: str) -> SentimentData:
        """Get sentiment from News API"""
        
        if not self.news_api_key:
            raise ValueError("News API key not configured")
        
        # Get company name for better search
        company_names = {
            'AAPL': 'Apple',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'META': 'Meta Facebook',
            'AMZN': 'Amazon',
            'SPY': 'S&P 500'
        }
        
        company = company_names.get(symbol, symbol)
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'"{company}" OR "{symbol}" stock market',
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 50,
            'from': (now_utc() - timedelta(days=1)).strftime('%Y-%m-%d')
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                articles = data.get('articles', [])
                
                # Analyze headlines and descriptions
                sentiment_scores = []
                keywords_found = []
                
                for article in articles:
                    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                    
                    # Enhanced news-specific keywords
                    positive_news = ['upgrade', 'beat', 'exceed', 'surge', 'rally', 'gain', 'profit', 'breakthrough']
                    negative_news = ['downgrade', 'miss', 'fall', 'plunge', 'loss', 'concern', 'investigation', 'lawsuit']
                    
                    positive_count = sum(1 for word in positive_news if word in text)
                    negative_count = sum(1 for word in negative_news if word in text)
                    
                    if positive_count > negative_count:
                        sentiment_scores.append(0.8)
                        keywords_found.extend([w for w in positive_news if w in text])
                    elif negative_count > positive_count:
                        sentiment_scores.append(-0.8)
                        keywords_found.extend([w for w in negative_news if w in text])
                    else:
                        sentiment_scores.append(0.0)
                
                # Calculate average sentiment
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
                
                return SentimentData(
                    source=SentimentSource.NEWS,
                    symbol=symbol,
                    score=np.clip(avg_sentiment, -1, 1),
                    confidence=min(len(articles) / 20, 1.0),
                    volume=len(articles),
                    timestamp=now_utc(),
                    raw_data={"article_count": len(articles)},
                    keywords=list(set(keywords_found))[:10]
                )
            else:
                logger.error(f"News API error: {response.status}")
                raise Exception(f"News API error: {response.status}")
    
    async def _get_reddit_auth_token(self) -> str:
        """Get Reddit OAuth token"""
        if self.reddit_token and self.reddit_token_expiry and now_utc() < self.reddit_token_expiry:
            return self.reddit_token
        
        # Get new token
        auth = aiohttp.BasicAuth(self.reddit_client_id, self.reddit_client_secret)
        data = {
            'grant_type': 'client_credentials',
            'scope': 'read'
        }
        headers = {'User-Agent': 'GoldenSignalsAI/1.0'}
        
        async with self.session.post(
            'https://www.reddit.com/api/v1/access_token',
            auth=auth,
            data=data,
            headers=headers
        ) as response:
            if response.status == 200:
                token_data = await response.json()
                self.reddit_token = token_data['access_token']
                self.reddit_token_expiry = now_utc() + timedelta(seconds=token_data['expires_in'] - 60)
                return self.reddit_token
            else:
                raise Exception(f"Reddit auth error: {response.status}")
    
    @error_recovery.with_recovery(
        fallback=lambda self, symbol: SentimentData(
            source=SentimentSource.REDDIT,
            symbol=symbol,
            score=0.0,
            confidence=0.0,
            volume=0,
            timestamp=now_utc(),
            raw_data={},
            keywords=[]
        ),
        severity=ErrorSeverity.MEDIUM
    )
    async def _get_reddit_sentiment(self, symbol: str) -> SentimentData:
        """Get sentiment from Reddit (WSB and other finance subreddits)"""
        
        if not self.reddit_client_id:
            raise ValueError("Reddit credentials not configured")
        
        # Get auth token
        token = await self._get_reddit_auth_token()
        
        headers = {
            'Authorization': f'Bearer {token}',
            'User-Agent': 'GoldenSignalsAI/1.0'
        }
        
        # Search multiple subreddits
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
        all_posts = []
        
        for subreddit in subreddits:
            url = f'https://oauth.reddit.com/r/{subreddit}/search'
            params = {
                'q': symbol,
                'sort': 'hot',
                'limit': 25,
                't': 'day',
                'restrict_sr': 'true'
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = data.get('data', {}).get('children', [])
                    all_posts.extend(posts)
        
        # Analyze sentiment
        sentiment_scores = []
        total_score = 0
        keywords_found = []
        
        for post in all_posts:
            post_data = post.get('data', {})
            title = post_data.get('title', '').lower()
            selftext = post_data.get('selftext', '').lower()
            text = f"{title} {selftext}"
            score = post_data.get('score', 0)
            num_comments = post_data.get('num_comments', 0)
            
            # Reddit-specific sentiment
            bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text)
            bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text)
            
            # Weight by engagement (upvotes + comments)
            engagement = score + num_comments * 2
            
            if bullish_count > bearish_count:
                sentiment_scores.append((0.8, engagement))
                keywords_found.extend([k for k in self.bullish_keywords if k in text])
            elif bearish_count > bullish_count:
                sentiment_scores.append((-0.8, engagement))
                keywords_found.extend([k for k in self.bearish_keywords if k in text])
            else:
                sentiment_scores.append((0.0, engagement))
            
            total_score += engagement
        
        # Calculate weighted sentiment
        if sentiment_scores and total_score > 0:
            weighted_sentiment = sum(
                score * engagement for score, engagement in sentiment_scores
            ) / total_score
        else:
            weighted_sentiment = 0.0
        
        return SentimentData(
            source=SentimentSource.REDDIT,
            symbol=symbol,
            score=np.clip(weighted_sentiment, -1, 1),
            confidence=min(len(all_posts) / 50, 1.0),
            volume=len(all_posts),
            timestamp=now_utc(),
            raw_data={"post_count": len(all_posts), "total_score": total_score},
            keywords=list(set(keywords_found))[:10]
        )
    
    @error_recovery.with_recovery(
        fallback=lambda self, symbol: SentimentData(
            source=SentimentSource.STOCKTWITS,
            symbol=symbol,
            score=0.0,
            confidence=0.0,
            volume=0,
            timestamp=now_utc(),
            raw_data={},
            keywords=[]
        ),
        severity=ErrorSeverity.LOW
    )
    async def _get_stocktwits_sentiment(self, symbol: str) -> SentimentData:
        """Get sentiment from StockTwits (no API key required)"""
        
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                messages = data.get('messages', [])
                
                # StockTwits provides sentiment labels
                bullish_count = 0
                bearish_count = 0
                total_likes = 0
                
                for message in messages:
                    sentiment = message.get('entities', {}).get('sentiment', {})
                    basic_sentiment = sentiment.get('basic', '')
                    likes = message.get('likes', {}).get('total', 0)
                    
                    if basic_sentiment == 'Bullish':
                        bullish_count += 1
                    elif basic_sentiment == 'Bearish':
                        bearish_count += 1
                    
                    total_likes += likes
                
                # Calculate sentiment ratio
                total_sentiment = bullish_count + bearish_count
                if total_sentiment > 0:
                    sentiment_score = (bullish_count - bearish_count) / total_sentiment
                else:
                    sentiment_score = 0.0
                
                return SentimentData(
                    source=SentimentSource.STOCKTWITS,
                    symbol=symbol,
                    score=np.clip(sentiment_score, -1, 1),
                    confidence=min(len(messages) / 30, 1.0),
                    volume=len(messages),
                    timestamp=now_utc(),
                    raw_data={
                        "bullish": bullish_count,
                        "bearish": bearish_count,
                        "total_likes": total_likes
                    },
                    keywords=[]
                )
            else:
                logger.error(f"StockTwits API error: {response.status}")
                raise Exception(f"StockTwits API error: {response.status}")
    
    async def get_market_sentiment_heatmap(
        self,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Get sentiment heatmap for multiple symbols"""
        
        tasks = [self.get_aggregated_sentiment(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        heatmap_data = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, dict):
                heatmap_data.append({
                    'symbol': symbol,
                    'sentiment': result.get('overall_score', 0),
                    'confidence': result.get('overall_confidence', 0),
                    'label': result.get('overall_label', 'Unknown'),
                    'volume': result.get('volume', {}).get('total', 0)
                })
            else:
                logger.error(f"Error getting sentiment for {symbol}: {result}")
                heatmap_data.append({
                    'symbol': symbol,
                    'sentiment': 0,
                    'confidence': 0,
                    'label': 'Error',
                    'volume': 0
                })
        
        # Sort by sentiment score
        heatmap_data.sort(key=lambda x: x['sentiment'], reverse=True)
        
        return {
            'heatmap': heatmap_data,
            'timestamp': now_utc().isoformat(),
            'summary': {
                'most_bullish': heatmap_data[0] if heatmap_data else None,
                'most_bearish': heatmap_data[-1] if heatmap_data else None,
                'average_sentiment': np.mean([d['sentiment'] for d in heatmap_data]) if heatmap_data else 0
            }
        }


# Singleton instance
enhanced_sentiment_service = EnhancedSentimentService()
