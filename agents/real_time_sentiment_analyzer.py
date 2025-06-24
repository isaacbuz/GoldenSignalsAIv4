"""
Real-time Sentiment Analyzer
Multi-source sentiment aggregation from social media, news, and forums
Issue #184: Agent-1: Develop Real-time Sentiment Analyzer
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
import logging
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Sources of sentiment data"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    NEWS = "news"
    FORUMS = "forums"
    DISCORD = "discord"
    TELEGRAM = "telegram"


class SentimentType(Enum):
    """Types of sentiment signals"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    FEAR = "fear"
    GREED = "greed"
    UNCERTAINTY = "uncertainty"


class InfluencerTier(Enum):
    """Tiers of social media influencers"""
    WHALE = "whale"  # >100k followers, high engagement
    INFLUENCER = "influencer"  # 10k-100k followers
    MICRO = "micro"  # 1k-10k followers
    RETAIL = "retail"  # <1k followers


@dataclass
class SentimentSignal:
    """Individual sentiment signal from a source"""
    id: str
    timestamp: datetime
    source: SentimentSource
    symbol: str
    content: str
    author: str
    author_tier: InfluencerTier
    followers: int
    engagement_rate: float
    sentiment_score: float  # -1 to 1
    sentiment_type: SentimentType
    confidence: float
    viral_score: float  # 0-100
    # Metadata
    likes: int
    shares: int
    comments: int
    reach: int
    # Analysis
    keywords: List[str]
    price_targets: List[float]
    timeframe: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source.value,
            'symbol': self.symbol,
            'author': self.author,
            'author_tier': self.author_tier.value,
            'sentiment_score': self.sentiment_score,
            'sentiment_type': self.sentiment_type.value,
            'confidence': self.confidence,
            'viral_score': self.viral_score,
            'engagement': {
                'likes': self.likes,
                'shares': self.shares,
                'comments': self.comments,
                'reach': self.reach
            }
        }


class SentimentAnalyzer:
    """Analyzes sentiment from text and metadata"""
    
    def __init__(self):
        # Sentiment keywords and patterns
        self.bullish_keywords = [
            'moon', 'rocket', 'bull', 'buy', 'long', 'calls', 'breakout',
            'squeeze', 'gamma', 'tendies', 'diamond hands', 'hodl', 'ath',
            'support', 'oversold', 'accumulation', 'reversal', 'bottom'
        ]
        
        self.bearish_keywords = [
            'crash', 'dump', 'bear', 'sell', 'short', 'puts', 'breakdown',
            'top', 'overbought', 'distribution', 'resistance', 'correction',
            'bubble', 'overvalued', 'red', 'bloodbath', 'capitulation'
        ]
        
        self.fear_keywords = [
            'panic', 'fear', 'worried', 'concern', 'risk', 'danger',
            'collapse', 'recession', 'depression', 'margin call'
        ]
        
        self.greed_keywords = [
            'fomo', 'yolo', 'all in', 'leverage', 'lambo', 'millionaire',
            'guaranteed', 'easy money', 'free money', 'cant lose'
        ]
        
        # Emoji sentiment mapping
        self.emoji_sentiment = {
            '🚀': 1.0, '🌙': 0.8, '📈': 0.7, '💎': 0.8, '🙌': 0.6,
            '💰': 0.7, '🤑': 0.8, '🔥': 0.6, '⬆️': 0.5, '✅': 0.5,
            '📉': -0.7, '🐻': -0.8, '💩': -0.9, '☠️': -0.9, '🩸': -0.8,
            '⬇️': -0.5, '❌': -0.5, '⚠️': -0.3, '😱': -0.6, '😭': -0.7
        }
    
    def analyze_text_sentiment(self, text: str) -> Tuple[float, SentimentType, float]:
        """Analyze sentiment from text content"""
        text_lower = text.lower()
        
        # Count keyword occurrences
        bullish_count = sum(1 for kw in self.bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in self.bearish_keywords if kw in text_lower)
        fear_count = sum(1 for kw in self.fear_keywords if kw in text_lower)
        greed_count = sum(1 for kw in self.greed_keywords if kw in text_lower)
        
        # Emoji sentiment
        emoji_score = sum(self.emoji_sentiment.get(char, 0) for char in text)
        
        # Calculate base sentiment
        sentiment_score = (bullish_count - bearish_count) / max(1, bullish_count + bearish_count)
        sentiment_score += emoji_score * 0.2  # Weight emoji contribution
        sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
        
        # Determine sentiment type
        if fear_count > max(bullish_count, bearish_count):
            sentiment_type = SentimentType.FEAR
        elif greed_count > max(bullish_count, bearish_count):
            sentiment_type = SentimentType.GREED
        elif sentiment_score > 0.3:
            sentiment_type = SentimentType.BULLISH
        elif sentiment_score < -0.3:
            sentiment_type = SentimentType.BEARISH
        else:
            sentiment_type = SentimentType.NEUTRAL
        
        # Confidence based on signal strength
        confidence = abs(sentiment_score) * 0.7 + min(emoji_score, 0.3)
        
        return sentiment_score, sentiment_type, confidence
    
    def extract_price_targets(self, text: str) -> List[float]:
        """Extract price targets from text"""
        # Pattern for price mentions ($XXX, XXX$, PT XXX)
        patterns = [
            r'\$(\d+(?:\.\d+)?)',  # $123.45
            r'(\d+(?:\.\d+)?)\$',  # 123.45$
            r'PT\s*(\d+(?:\.\d+)?)',  # PT 123.45
            r'target\s*(?:of\s*)?(\d+(?:\.\d+)?)',  # target 123.45
            r'(\d+(?:\.\d+)?)\s*(?:price\s*)?target'  # 123.45 target
        ]
        
        targets = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            targets.extend(float(match) for match in matches)
        
        # Filter reasonable targets (remove outliers)
        if targets:
            median = statistics.median(targets)
            reasonable_targets = [t for t in targets if 0.5 * median <= t <= 2 * median]
            return reasonable_targets[:3]  # Return top 3
        
        return []
    
    def calculate_viral_score(self, signal: Dict[str, Any]) -> float:
        """Calculate how viral a post is"""
        engagement = signal.get('likes', 0) + signal.get('shares', 0) * 2 + signal.get('comments', 0)
        followers = max(1, signal.get('followers', 1))
        
        # Engagement rate
        engagement_rate = engagement / followers
        
        # Time decay (newer = higher score)
        hours_old = (datetime.now() - signal.get('timestamp', datetime.now())).total_seconds() / 3600
        time_factor = max(0, 1 - hours_old / 24)  # Decay over 24 hours
        
        # Viral score calculation
        viral_score = min(100, engagement_rate * 1000 * time_factor)
        
        # Boost for high absolute engagement
        if engagement > 10000:
            viral_score = min(100, viral_score * 1.5)
        elif engagement > 1000:
            viral_score = min(100, viral_score * 1.2)
        
        return viral_score


class RealTimeSentimentAnalyzer:
    """
    Real-time sentiment analyzer aggregating multiple sources
    Provides weighted sentiment signals based on influencer tiers and virality
    """
    
    def __init__(self, use_mock_data: bool = True):
        """Initialize the sentiment analyzer"""
        self.use_mock_data = use_mock_data
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sentiment_signals: List[SentimentSignal] = []
        self.aggregated_sentiment: Dict[str, Dict[str, Any]] = {}
        
        # Weight configuration
        self.source_weights = {
            SentimentSource.TWITTER: 0.25,
            SentimentSource.REDDIT: 0.20,
            SentimentSource.STOCKTWITS: 0.20,
            SentimentSource.NEWS: 0.25,
            SentimentSource.FORUMS: 0.05,
            SentimentSource.DISCORD: 0.03,
            SentimentSource.TELEGRAM: 0.02
        }
        
        self.tier_weights = {
            InfluencerTier.WHALE: 1.0,
            InfluencerTier.INFLUENCER: 0.5,
            InfluencerTier.MICRO: 0.2,
            InfluencerTier.RETAIL: 0.1
        }
        
        if use_mock_data:
            self._load_mock_sentiment_data()
    
    def _load_mock_sentiment_data(self):
        """Load mock sentiment data for testing"""
        mock_signals = [
            # Twitter whale bullish on AAPL
            SentimentSignal(
                id="tw_whale_aapl_001",
                timestamp=datetime.now() - timedelta(hours=1),
                source=SentimentSource.TWITTER,
                symbol="AAPL",
                content="$AAPL 🚀🚀 Major breakout incoming! Loading calls here. PT $220 🌙",
                author="TechWhale",
                author_tier=InfluencerTier.WHALE,
                followers=150000,
                engagement_rate=0.05,
                sentiment_score=0.9,
                sentiment_type=SentimentType.BULLISH,
                confidence=0.85,
                viral_score=75.0,
                likes=7500,
                shares=2000,
                comments=500,
                reach=300000,
                keywords=['breakout', 'calls', 'moon'],
                price_targets=[220.0],
                timeframe="1-2 weeks"
            ),
            # Reddit DD post bearish on SPY
            SentimentSignal(
                id="reddit_dd_spy_001",
                timestamp=datetime.now() - timedelta(hours=3),
                source=SentimentSource.REDDIT,
                symbol="SPY",
                content="DD: SPY overextended, expecting correction to 430. VIX coiling, puts cheap",
                author="BearGang2023",
                author_tier=InfluencerTier.INFLUENCER,
                followers=25000,
                engagement_rate=0.08,
                sentiment_score=-0.7,
                sentiment_type=SentimentType.BEARISH,
                confidence=0.75,
                viral_score=60.0,
                likes=2000,
                shares=100,
                comments=300,
                reach=50000,
                keywords=['correction', 'puts', 'vix'],
                price_targets=[430.0],
                timeframe="1 week"
            ),
            # StockTwits retail FOMO
            SentimentSignal(
                id="st_retail_tsla_001",
                timestamp=datetime.now() - timedelta(minutes=30),
                source=SentimentSource.STOCKTWITS,
                symbol="TSLA",
                content="$TSLA YOLO'd my entire account! 🚀🚀🚀 $300 EOW guaranteed! 💎🙌",
                author="RetailTrader99",
                author_tier=InfluencerTier.RETAIL,
                followers=500,
                engagement_rate=0.02,
                sentiment_score=0.95,
                sentiment_type=SentimentType.GREED,
                confidence=0.4,
                viral_score=20.0,
                likes=10,
                shares=2,
                comments=5,
                reach=1000,
                keywords=['yolo', 'moon', 'diamond hands'],
                price_targets=[300.0],
                timeframe="EOW"
            ),
            # News sentiment
            SentimentSignal(
                id="news_aapl_001",
                timestamp=datetime.now() - timedelta(hours=2),
                source=SentimentSource.NEWS,
                symbol="AAPL",
                content="Apple announces record iPhone sales, beats earnings estimates by 15%",
                author="FinancialTimes",
                author_tier=InfluencerTier.WHALE,
                followers=2000000,
                engagement_rate=0.01,
                sentiment_score=0.8,
                sentiment_type=SentimentType.BULLISH,
                confidence=0.9,
                viral_score=85.0,
                likes=50000,
                shares=10000,
                comments=2000,
                reach=5000000,
                keywords=['earnings', 'beat', 'record'],
                price_targets=[],
                timeframe="quarterly"
            ),
            # Fear signal
            SentimentSignal(
                id="twitter_fear_spy_001",
                timestamp=datetime.now() - timedelta(minutes=15),
                source=SentimentSource.TWITTER,
                symbol="SPY",
                content="⚠️ BREAKING: Fed emergency meeting scheduled. Market crash incoming? 😱📉",
                author="MarketPanic",
                author_tier=InfluencerTier.MICRO,
                followers=5000,
                engagement_rate=0.15,
                sentiment_score=-0.9,
                sentiment_type=SentimentType.FEAR,
                confidence=0.6,
                viral_score=45.0,
                likes=750,
                shares=200,
                comments=100,
                reach=15000,
                keywords=['crash', 'fear', 'panic'],
                price_targets=[],
                timeframe="immediate"
            )
        ]
        
        self.sentiment_signals.extend(mock_signals)
    
    async def analyze_symbol_sentiment(self, symbol: str, 
                                     timeframe: str = '1h') -> Dict[str, Any]:
        """Analyze aggregated sentiment for a symbol"""
        # Filter signals for symbol and timeframe
        cutoff_time = self._get_cutoff_time(timeframe)
        relevant_signals = [
            s for s in self.sentiment_signals 
            if s.symbol == symbol and s.timestamp >= cutoff_time
        ]
        
        if not relevant_signals:
            return {
                'symbol': symbol,
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'signal_count': 0,
                'message': 'No recent sentiment data available'
            }
        
        # Aggregate sentiment with weights
        weighted_sentiment = 0.0
        total_weight = 0.0
        sentiment_breakdown = defaultdict(float)
        source_breakdown = defaultdict(int)
        
        for signal in relevant_signals:
            # Calculate signal weight
            source_weight = self.source_weights.get(signal.source, 0.1)
            tier_weight = self.tier_weights.get(signal.author_tier, 0.1)
            viral_weight = signal.viral_score / 100
            
            # Combined weight
            signal_weight = source_weight * tier_weight * (0.5 + 0.5 * viral_weight)
            
            # Add to aggregation
            weighted_sentiment += signal.sentiment_score * signal_weight
            total_weight += signal_weight
            
            # Track breakdowns
            sentiment_breakdown[signal.sentiment_type] += 1
            source_breakdown[signal.source] += 1
        
        # Calculate final sentiment
        final_sentiment_score = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Determine sentiment direction
        if final_sentiment_score > 0.3:
            sentiment_direction = 'bullish'
        elif final_sentiment_score < -0.3:
            sentiment_direction = 'bearish'
        else:
            sentiment_direction = 'neutral'
        
        # Calculate confidence
        signal_diversity = len(set(s.source for s in relevant_signals))
        confidence = min(0.9, (len(relevant_signals) / 10) * (signal_diversity / 4))
        
        # Extract key insights
        insights = self._extract_insights(relevant_signals)
        
        # Get top influencers
        top_signals = sorted(relevant_signals, 
                           key=lambda s: s.viral_score * self.tier_weights.get(s.author_tier, 0.1), 
                           reverse=True)[:3]
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'sentiment': sentiment_direction,
            'score': float(final_sentiment_score),
            'confidence': float(confidence),
            'signal_count': len(relevant_signals),
            'sentiment_breakdown': dict(sentiment_breakdown),
            'source_breakdown': dict(source_breakdown),
            'top_influencers': [
                {
                    'author': s.author,
                    'source': s.source.value,
                    'sentiment': s.sentiment_type.value,
                    'viral_score': s.viral_score,
                    'content_preview': s.content[:100] + '...' if len(s.content) > 100 else s.content
                }
                for s in top_signals
            ],
            'insights': insights,
            'trading_signal': self._generate_trading_signal(
                final_sentiment_score, confidence, sentiment_breakdown, insights
            ),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_cutoff_time(self, timeframe: str) -> datetime:
        """Get cutoff time based on timeframe"""
        now = datetime.now()
        if timeframe == '15m':
            return now - timedelta(minutes=15)
        elif timeframe == '1h':
            return now - timedelta(hours=1)
        elif timeframe == '4h':
            return now - timedelta(hours=4)
        elif timeframe == '1d':
            return now - timedelta(days=1)
        else:
            return now - timedelta(hours=1)
    
    def _extract_insights(self, signals: List[SentimentSignal]) -> List[str]:
        """Extract key insights from signals"""
        insights = []
        
        # Check for consensus
        bullish_count = sum(1 for s in signals if s.sentiment_type == SentimentType.BULLISH)
        bearish_count = sum(1 for s in signals if s.sentiment_type == SentimentType.BEARISH)
        
        if bullish_count > bearish_count * 2:
            insights.append("Strong bullish consensus across sources")
        elif bearish_count > bullish_count * 2:
            insights.append("Strong bearish consensus across sources")
        
        # Check for whale activity
        whale_signals = [s for s in signals if s.author_tier == InfluencerTier.WHALE]
        if whale_signals:
            whale_sentiment = np.mean([s.sentiment_score for s in whale_signals])
            if whale_sentiment > 0.5:
                insights.append("Major influencers are bullish")
            elif whale_sentiment < -0.5:
                insights.append("Major influencers are bearish")
        
        # Check for fear/greed extremes
        fear_signals = [s for s in signals if s.sentiment_type == SentimentType.FEAR]
        greed_signals = [s for s in signals if s.sentiment_type == SentimentType.GREED]
        
        if len(fear_signals) > len(signals) * 0.3:
            insights.append("High fear detected - potential buying opportunity")
        if len(greed_signals) > len(signals) * 0.3:
            insights.append("Extreme greed detected - caution advised")
        
        # Price target consensus
        all_targets = []
        for s in signals:
            all_targets.extend(s.price_targets)
        
        if all_targets:
            avg_target = np.mean(all_targets)
            insights.append(f"Average price target: ${avg_target:.2f}")
        
        return insights[:5]  # Top 5 insights
    
    def _generate_trading_signal(self, sentiment_score: float, confidence: float,
                               sentiment_breakdown: Dict, insights: List[str]) -> Dict[str, Any]:
        """Generate trading signal based on sentiment analysis"""
        signal = {
            'action': 'hold',
            'strength': 'weak',
            'timeframe': 'short_term',
            'risk_level': 'medium',
            'entry_strategy': None,
            'position_size': 0.5
        }
        
        # Strong bullish sentiment
        if sentiment_score > 0.6 and confidence > 0.7:
            signal['action'] = 'buy'
            signal['strength'] = 'strong'
            signal['entry_strategy'] = 'scale_in_on_dips'
            signal['position_size'] = 0.75
        elif sentiment_score > 0.3 and confidence > 0.5:
            signal['action'] = 'buy'
            signal['strength'] = 'moderate'
            signal['entry_strategy'] = 'wait_for_confirmation'
            signal['position_size'] = 0.5
        
        # Strong bearish sentiment
        elif sentiment_score < -0.6 and confidence > 0.7:
            signal['action'] = 'sell'
            signal['strength'] = 'strong'
            signal['entry_strategy'] = 'immediate_exit'
            signal['position_size'] = 0.0
        elif sentiment_score < -0.3 and confidence > 0.5:
            signal['action'] = 'reduce'
            signal['strength'] = 'moderate'
            signal['entry_strategy'] = 'scale_out'
            signal['position_size'] = 0.25
        
        # Adjust for fear/greed extremes
        if sentiment_breakdown.get(SentimentType.FEAR, 0) > sentiment_breakdown.get(SentimentType.BULLISH, 0):
            signal['risk_level'] = 'high'
            if signal['action'] == 'buy':
                signal['entry_strategy'] = 'contrarian_buy'
        
        if sentiment_breakdown.get(SentimentType.GREED, 0) > sentiment_breakdown.get(SentimentType.BEARISH, 0):
            signal['risk_level'] = 'very_high'
            if signal['action'] == 'buy':
                signal['position_size'] *= 0.5  # Reduce size in greed
        
        return signal
    
    async def get_trending_symbols(self, min_signals: int = 5) -> List[Dict[str, Any]]:
        """Get symbols with high sentiment activity"""
        symbol_activity = defaultdict(lambda: {'count': 0, 'viral_score': 0, 'sentiment': 0})
        
        # Aggregate by symbol
        for signal in self.sentiment_signals:
            symbol = signal.symbol
            symbol_activity[symbol]['count'] += 1
            symbol_activity[symbol]['viral_score'] += signal.viral_score
            symbol_activity[symbol]['sentiment'] += signal.sentiment_score
        
        # Filter and sort
        trending = []
        for symbol, data in symbol_activity.items():
            if data['count'] >= min_signals:
                avg_sentiment = data['sentiment'] / data['count']
                avg_viral = data['viral_score'] / data['count']
                
                trending.append({
                    'symbol': symbol,
                    'signal_count': data['count'],
                    'avg_viral_score': avg_viral,
                    'sentiment': 'bullish' if avg_sentiment > 0.2 else 'bearish' if avg_sentiment < -0.2 else 'neutral',
                    'sentiment_score': avg_sentiment,
                    'momentum': avg_viral * abs(avg_sentiment)  # Combined metric
                })
        
        # Sort by momentum
        trending.sort(key=lambda x: x['momentum'], reverse=True)
        return trending[:10]  # Top 10
    
    async def monitor_sentiment_changes(self, symbol: str, 
                                      interval_minutes: int = 15) -> Dict[str, Any]:
        """Monitor sentiment changes over time"""
        current = await self.analyze_symbol_sentiment(symbol, '15m')
        previous = await self.analyze_symbol_sentiment(symbol, '1h')
        
        # Calculate changes
        sentiment_change = current['score'] - previous['score']
        signal_change = current['signal_count'] - previous['signal_count']
        
        # Determine if significant shift
        significant_shift = abs(sentiment_change) > 0.3 or abs(signal_change) > 10
        
        return {
            'symbol': symbol,
            'current_sentiment': current['sentiment'],
            'sentiment_change': sentiment_change,
            'signal_change': signal_change,
            'significant_shift': significant_shift,
            'shift_direction': 'bullish' if sentiment_change > 0 else 'bearish' if sentiment_change < 0 else 'stable',
            'alert_level': 'high' if significant_shift else 'low',
            'recommendation': self._generate_shift_recommendation(
                sentiment_change, signal_change, current, previous
            )
        }
    
    def _generate_shift_recommendation(self, sentiment_change: float, 
                                     signal_change: int,
                                     current: Dict, previous: Dict) -> str:
        """Generate recommendation based on sentiment shift"""
        if sentiment_change > 0.5 and signal_change > 5:
            return "Strong bullish momentum building - consider entering positions"
        elif sentiment_change < -0.5 and signal_change > 5:
            return "Bearish sentiment surge - consider protective measures"
        elif abs(sentiment_change) < 0.1:
            return "Sentiment stable - maintain current positions"
        else:
            return "Monitor closely for confirmation of trend"


# Demo function
async def demo_sentiment_analyzer():
    """Demonstrate the Real-time Sentiment Analyzer"""
    analyzer = RealTimeSentimentAnalyzer(use_mock_data=True)
    
    print("Real-time Sentiment Analyzer Demo")
    print("="*70)
    
    # Test 1: Analyze AAPL sentiment
    print("\n1. Analyzing AAPL Sentiment:")
    print("-"*50)
    
    aapl_sentiment = await analyzer.analyze_symbol_sentiment('AAPL', '4h')  # Use 4h to capture more signals
    
    print(f"Overall Sentiment: {aapl_sentiment['sentiment'].upper()}")
    print(f"Sentiment Score: {aapl_sentiment['score']:.2f} ({aapl_sentiment['confidence']:.1%} confidence)")
    print(f"Signal Count: {aapl_sentiment['signal_count']}")
    
    if 'top_influencers' in aapl_sentiment and aapl_sentiment['top_influencers']:
        print(f"\nTop Influencers:")
        for inf in aapl_sentiment['top_influencers']:
            print(f"  - {inf['author']} ({inf['source']}): {inf['sentiment']}")
            print(f"    Viral Score: {inf['viral_score']:.0f}")
            print(f"    \"{inf['content_preview']}\"")
    
    print(f"\nKey Insights:")
    for insight in aapl_sentiment['insights']:
        print(f"  • {insight}")
    
    print(f"\nTrading Signal:")
    signal = aapl_sentiment['trading_signal']
    print(f"  Action: {signal['action'].upper()}")
    print(f"  Strength: {signal['strength']}")
    print(f"  Position Size: {signal['position_size']:.1%}")
    
    # Test 2: Get trending symbols
    print("\n\n2. Trending Symbols by Sentiment:")
    print("-"*50)
    
    trending = await analyzer.get_trending_symbols(min_signals=1)
    
    for i, symbol in enumerate(trending[:5], 1):
        print(f"\n{i}. {symbol['symbol']}")
        print(f"   Sentiment: {symbol['sentiment'].upper()} ({symbol['sentiment_score']:.2f})")
        print(f"   Signal Count: {symbol['signal_count']}")
        print(f"   Viral Score: {symbol['avg_viral_score']:.0f}")
        print(f"   Momentum: {symbol['momentum']:.1f}")
    
    # Test 3: Monitor sentiment changes
    print("\n\n3. Monitoring Sentiment Changes:")
    print("-"*50)
    
    for symbol in ['AAPL', 'SPY', 'TSLA']:
        changes = await analyzer.monitor_sentiment_changes(symbol)
        
        print(f"\n{symbol}:")
        print(f"  Current: {changes['current_sentiment'].upper()}")
        print(f"  Change: {changes['sentiment_change']:+.2f}")
        print(f"  Shift: {changes['shift_direction'].upper()}")
        print(f"  Alert: {changes['alert_level'].upper()}")
        print(f"  📋 {changes['recommendation']}")


if __name__ == "__main__":
    asyncio.run(demo_sentiment_analyzer()) 