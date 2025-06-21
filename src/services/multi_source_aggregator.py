"""
Multi-Source Data Aggregator for GoldenSignalsAI
Integrates multiple data sources for comprehensive market intelligence
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import logging
import os
from abc import ABC, abstractmethod

# Import for various APIs
import yfinance as yf
import pyEX
import requests
from fredapi import Fred
import praw  # Reddit API
import tweepy  # Twitter API
import discord
from textblob import TextBlob
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """Configuration for a data source"""
    name: str
    api_key: Optional[str]
    rate_limit: int  # requests per minute
    weight: float  # importance weight for aggregation
    cache_ttl: int  # cache time-to-live in seconds


@dataclass
class MarketSentiment:
    """Aggregated market sentiment data"""
    symbol: str
    timestamp: datetime
    composite_score: float  # -1 to 1
    confidence: float  # 0 to 1
    source_scores: Dict[str, float]
    volume_indicators: Dict[str, Any]
    news_count: int
    social_mentions: int
    options_flow: Optional[Dict[str, Any]]


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    def __init__(self, config: SourceConfig):
        self.config = config
        self.last_request_time = None
        self.cache = {}
    
    @abstractmethod
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment data for a symbol"""
        pass
    
    async def rate_limit_check(self):
        """Ensure we don't exceed rate limits"""
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            min_interval = 60 / self.config.rate_limit
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
        self.last_request_time = datetime.now()


class IEXCloudSource(DataSource):
    """IEX Cloud data source"""
    
    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self.client = pyEX.Client(api_token=config.api_key, version='stable')
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        await self.rate_limit_check()
        
        try:
            # Get quote and news
            quote = self.client.quote(symbol)
            news = self.client.news(symbol, last=10)
            
            # Calculate sentiment from news headlines
            news_sentiment = 0
            if news:
                sentiments = [TextBlob(article['headline']).sentiment.polarity for article in news]
                news_sentiment = np.mean(sentiments) if sentiments else 0
            
            # Price momentum
            price_change = quote.get('changePercent', 0)
            
            return {
                'sentiment': news_sentiment,
                'price_momentum': price_change,
                'volume': quote.get('volume', 0),
                'news_count': len(news),
                'market_cap': quote.get('marketCap', 0)
            }
        except Exception as e:
            logger.error(f"IEX Cloud error for {symbol}: {e}")
            return {}


class StockTwitsSource(DataSource):
    """StockTwits social sentiment"""
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        await self.rate_limit_check()
        
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
            
            messages = data.get('messages', [])
            
            # Count bullish/bearish
            bullish = 0
            bearish = 0
            
            for msg in messages:
                sentiment = msg.get('entities', {}).get('sentiment', {}).get('basic')
                if sentiment == 'Bullish':
                    bullish += 1
                elif sentiment == 'Bearish':
                    bearish += 1
            
            total = bullish + bearish
            sentiment_score = (bullish - bearish) / total if total > 0 else 0
            
            return {
                'sentiment': sentiment_score,
                'bullish_count': bullish,
                'bearish_count': bearish,
                'message_count': len(messages),
                'trending_score': data.get('symbol', {}).get('trending_score', 0)
            }
        except Exception as e:
            logger.error(f"StockTwits error for {symbol}: {e}")
            return {}


class FREDSource(DataSource):
    """Federal Reserve Economic Data"""
    
    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self.fred = Fred(api_key=config.api_key)
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get macro indicators that might affect the symbol"""
        await self.rate_limit_check()
        
        try:
            # Get key macro indicators
            vix = self.fred.get_series_latest_release('VIXCLS').iloc[-1]
            unemployment = self.fred.get_series_latest_release('UNRATE').iloc[-1]
            
            # Calculate macro sentiment
            # Lower VIX and unemployment = positive sentiment
            vix_sentiment = (30 - vix) / 30  # Normalize VIX
            unemployment_sentiment = (10 - unemployment) / 10
            
            macro_sentiment = (vix_sentiment + unemployment_sentiment) / 2
            
            return {
                'macro_sentiment': macro_sentiment,
                'vix': vix,
                'unemployment': unemployment,
                'fear_greed': vix_sentiment
            }
        except Exception as e:
            logger.error(f"FRED error: {e}")
            return {}


class RedditSource(DataSource):
    """Reddit sentiment analysis"""
    
    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='GoldenSignalsAI/1.0'
        )
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        await self.rate_limit_check()
        
        try:
            # Search multiple subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'options']
            
            mentions = 0
            sentiments = []
            
            for sub_name in subreddits:
                subreddit = self.reddit.subreddit(sub_name)
                
                # Search for symbol mentions in hot posts
                for submission in subreddit.hot(limit=25):
                    if symbol.upper() in submission.title.upper() or \
                       f"${symbol.upper()}" in submission.title:
                        mentions += 1
                        
                        # Analyze sentiment
                        blob = TextBlob(submission.title)
                        sentiments.append(blob.sentiment.polarity)
                        
                        # Check top comments
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments[:5]:
                            if hasattr(comment, 'body'):
                                sentiments.append(TextBlob(comment.body).sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                'sentiment': avg_sentiment,
                'mention_count': mentions,
                'wsb_heat': mentions / 100,  # Normalize to 0-1
                'retail_interest': len(sentiments)
            }
        except Exception as e:
            logger.error(f"Reddit error for {symbol}: {e}")
            return {}


class DiscordMonitor(DataSource):
    """Discord trading community monitor"""
    
    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self.recent_messages = {}
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        # This would connect to Discord bot
        # For now, return mock data
        return {
            'sentiment': 0.2,
            'mention_velocity': 5,  # mentions per hour
            'pump_probability': 0.3
        }


class MultiSourceAggregator:
    """Aggregates data from multiple sources"""
    
    def __init__(self):
        self.sources = self._initialize_sources()
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _initialize_sources(self) -> Dict[str, DataSource]:
        """Initialize all data sources"""
        sources = {}
        
        # IEX Cloud
        if os.getenv('IEX_API_KEY'):
            sources['iex'] = IEXCloudSource(SourceConfig(
                name='IEX Cloud',
                api_key=os.getenv('IEX_API_KEY'),
                rate_limit=100,
                weight=0.3,
                cache_ttl=60
            ))
        
        # StockTwits
        sources['stocktwits'] = StockTwitsSource(SourceConfig(
            name='StockTwits',
            api_key=None,
            rate_limit=200,
            weight=0.2,
            cache_ttl=300
        ))
        
        # FRED
        if os.getenv('FRED_API_KEY'):
            sources['fred'] = FREDSource(SourceConfig(
                name='FRED',
                api_key=os.getenv('FRED_API_KEY'),
                rate_limit=120,
                weight=0.1,
                cache_ttl=3600
            ))
        
        # Reddit
        if os.getenv('REDDIT_CLIENT_ID'):
            sources['reddit'] = RedditSource(SourceConfig(
                name='Reddit',
                api_key=None,
                rate_limit=60,
                weight=0.2,
                cache_ttl=600
            ))
        
        # Discord
        sources['discord'] = DiscordMonitor(SourceConfig(
            name='Discord',
            api_key=None,
            rate_limit=100,
            weight=0.2,
            cache_ttl=60
        ))
        
        return sources
    
    async def get_aggregated_sentiment(self, symbol: str) -> MarketSentiment:
        """Get aggregated sentiment from all sources"""
        
        # Check cache
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Gather data from all sources
        tasks = []
        source_names = []
        
        for name, source in self.sources.items():
            tasks.append(source.get_sentiment(symbol))
            source_names.append(name)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        source_scores = {}
        weighted_sum = 0
        total_weight = 0
        news_count = 0
        social_mentions = 0
        
        for name, result in zip(source_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error from {name}: {result}")
                continue
            
            if result and 'sentiment' in result:
                source = self.sources[name]
                score = result['sentiment']
                source_scores[name] = score
                
                weighted_sum += score * source.config.weight
                total_weight += source.config.weight
                
                # Aggregate metrics
                news_count += result.get('news_count', 0)
                social_mentions += result.get('mention_count', 0) + \
                                 result.get('message_count', 0)
        
        # Calculate composite score
        composite_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Calculate confidence based on source agreement
        if source_scores:
            scores = list(source_scores.values())
            confidence = 1 - np.std(scores)  # Higher agreement = higher confidence
        else:
            confidence = 0
        
        sentiment = MarketSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            composite_score=composite_score,
            confidence=confidence,
            source_scores=source_scores,
            volume_indicators={},
            news_count=news_count,
            social_mentions=social_mentions,
            options_flow=None
        )
        
        # Cache result
        self.cache[cache_key] = sentiment
        
        return sentiment
    
    async def get_market_pulse(self, symbols: List[str]) -> Dict[str, MarketSentiment]:
        """Get sentiment for multiple symbols"""
        tasks = [self.get_aggregated_sentiment(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return {
            sentiment.symbol: sentiment 
            for sentiment in results
        }


# Example usage
async def main():
    aggregator = MultiSourceAggregator()
    
    # Get sentiment for a single symbol
    sentiment = await aggregator.get_aggregated_sentiment('AAPL')
    print(f"AAPL Sentiment: {sentiment.composite_score:.2f} (confidence: {sentiment.confidence:.2f})")
    print(f"Sources: {sentiment.source_scores}")
    
    # Get market pulse
    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    market_pulse = await aggregator.get_market_pulse(symbols)
    
    for symbol, sentiment in market_pulse.items():
        print(f"{symbol}: {sentiment.composite_score:+.2f}")


if __name__ == "__main__":
    asyncio.run(main()) 