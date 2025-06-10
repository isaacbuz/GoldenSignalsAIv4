"""
Sentiment Analysis Agent - GoldenSignalsAI V3

Analyzes market sentiment from news articles, social media, and financial reports.
Uses NLP techniques to gauge bullish/bearish sentiment for trading signals.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import re
from loguru import logger

from .base import BaseAgent
from ..models.signals import Signal, SignalType, SignalStrength


class SentimentAnalysisAgent(BaseAgent):
    """
    Agent specializing in sentiment analysis from various text sources
    """
    
    def __init__(self, name: str, db_manager, redis_manager):
        super().__init__(name, db_manager, redis_manager)
        
        # Sentiment keywords and weights
        self.bullish_keywords = {
            "strong": 0.8, "bullish": 0.9, "positive": 0.6, "growth": 0.7,
            "increase": 0.6, "rise": 0.6, "up": 0.5, "gain": 0.7,
            "outperform": 0.8, "buy": 0.9, "upgrade": 0.8, "beat": 0.7,
            "exceed": 0.7, "momentum": 0.6, "rally": 0.8, "surge": 0.8
        }
        
        self.bearish_keywords = {
            "weak": 0.8, "bearish": 0.9, "negative": 0.6, "decline": 0.7,
            "decrease": 0.6, "fall": 0.6, "down": 0.5, "loss": 0.7,
            "underperform": 0.8, "sell": 0.9, "downgrade": 0.8, "miss": 0.7,
            "below": 0.6, "concern": 0.6, "drop": 0.7, "plunge": 0.8
        }
        
        # Mock news sources (in production, integrate with actual APIs)
        self.news_sources = {
            "reuters": 0.9,
            "bloomberg": 0.9,
            "wsj": 0.8,
            "cnbc": 0.7,
            "marketwatch": 0.6,
            "seeking_alpha": 0.5,
            "reddit": 0.3,
            "twitter": 0.4
        }
    
    async def analyze(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Signal]:
        """
        Perform sentiment analysis for the given symbol
        
        Args:
            symbol: Stock symbol
            market_data: Current market data
            historical_data: Historical price data (not used for sentiment)
            
        Returns:
            Sentiment-based signal or None
        """
        try:
            # Get recent news and sentiment data
            news_data = await self._fetch_news_data(symbol)
            social_data = await self._fetch_social_media_data(symbol)
            
            # Analyze sentiment from various sources
            news_sentiment = await self._analyze_news_sentiment(news_data)
            social_sentiment = await self._analyze_social_sentiment(social_data)
            
            # Combine sentiment scores
            combined_sentiment = await self._combine_sentiment_scores(
                news_sentiment, social_sentiment
            )
            
            # Generate signal based on sentiment
            signal = await self._generate_sentiment_signal(
                symbol, market_data, combined_sentiment
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {str(e)}")
            return None
    
    async def _fetch_news_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch recent news articles about the symbol"""
        try:
            # Mock news data (in production, integrate with news APIs)
            # This would fetch from Reuters, Bloomberg, etc.
            
            current_price = await self._get_cached_price(symbol)
            
            # Generate mock news based on recent price movement
            mock_news = []
            if current_price:
                # Simulate news sentiment based on price
                change = (current_price - 100) / 100  # Mock baseline of 100
                
                if change > 0.05:  # Price up significantly
                    mock_news.append({
                        "title": f"{symbol} stock surges on strong earnings report",
                        "content": f"{symbol} showed strong performance with revenue growth exceeding expectations",
                        "source": "reuters",
                        "timestamp": datetime.utcnow() - timedelta(hours=2),
                        "sentiment_score": 0.8
                    })
                elif change < -0.05:  # Price down significantly
                    mock_news.append({
                        "title": f"{symbol} shares decline amid market concerns",
                        "content": f"{symbol} faces headwinds as investors express concerns about market conditions",
                        "source": "bloomberg",
                        "timestamp": datetime.utcnow() - timedelta(hours=1),
                        "sentiment_score": -0.7
                    })
                else:  # Neutral price movement
                    mock_news.append({
                        "title": f"{symbol} maintains steady performance",
                        "content": f"{symbol} shows stable trading patterns with mixed investor sentiment",
                        "source": "marketwatch",
                        "timestamp": datetime.utcnow() - timedelta(minutes=30),
                        "sentiment_score": 0.1
                    })
            
            return mock_news
            
        except Exception as e:
            logger.error(f"Failed to fetch news data for {symbol}: {str(e)}")
            return []
    
    async def _fetch_social_media_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch social media sentiment about the symbol"""
        try:
            # Mock social media data (in production, integrate with Twitter API, Reddit API, etc.)
            
            current_price = await self._get_cached_price(symbol)
            
            mock_social = []
            if current_price:
                # Generate mock social sentiment
                base_sentiment = (current_price % 10) / 10 - 0.5  # Random-ish sentiment
                
                mock_social.extend([
                    {
                        "platform": "twitter",
                        "content": f"Bullish on ${symbol} - great fundamentals!",
                        "mentions": 150,
                        "sentiment_score": base_sentiment + 0.3,
                        "timestamp": datetime.utcnow() - timedelta(hours=1)
                    },
                    {
                        "platform": "reddit",
                        "content": f"${symbol} analysis - mixed signals but overall positive",
                        "mentions": 45,
                        "sentiment_score": base_sentiment,
                        "timestamp": datetime.utcnow() - timedelta(hours=3)
                    },
                    {
                        "platform": "stocktwits",
                        "content": f"${symbol} breaking out - watching for continuation",
                        "mentions": 200,
                        "sentiment_score": base_sentiment + 0.2,
                        "timestamp": datetime.utcnow() - timedelta(minutes=45)
                    }
                ])
            
            return mock_social
            
        except Exception as e:
            logger.error(f"Failed to fetch social media data for {symbol}: {str(e)}")
            return []
    
    async def _get_cached_price(self, symbol: str) -> Optional[float]:
        """Get cached price for mock data generation"""
        try:
            # Try to get from Redis cache
            cached_quote = await self.redis_manager.get_cached_quote(symbol)
            if cached_quote:
                return cached_quote.get("price")
            
            # Return a mock price based on symbol hash for consistency
            return 100 + (hash(symbol) % 100)
            
        except Exception as e:
            logger.error(f"Failed to get cached price for {symbol}: {str(e)}")
            return None
    
    async def _analyze_news_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze sentiment from news articles"""
        try:
            if not news_data:
                return {"score": 0.0, "confidence": 0.0, "article_count": 0}
            
            total_sentiment = 0.0
            weighted_sentiment = 0.0
            total_weight = 0.0
            
            for article in news_data:
                content = f"{article.get('title', '')} {article.get('content', '')}".lower()
                source = article.get("source", "unknown")
                source_weight = self.news_sources.get(source, 0.5)
                
                # Calculate sentiment using keyword analysis
                sentiment_score = self._calculate_text_sentiment(content)
                
                # Weight by source credibility
                weighted_score = sentiment_score * source_weight
                weighted_sentiment += weighted_score
                total_weight += source_weight
                total_sentiment += sentiment_score
            
            avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
            confidence = min(len(news_data) / 10.0, 1.0)  # More articles = higher confidence
            
            return {
                "score": avg_sentiment,
                "confidence": confidence,
                "article_count": len(news_data),
                "raw_sentiment": total_sentiment / len(news_data) if news_data else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze news sentiment: {str(e)}")
            return {"score": 0.0, "confidence": 0.0, "article_count": 0}
    
    async def _analyze_social_sentiment(self, social_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze sentiment from social media"""
        try:
            if not social_data:
                return {"score": 0.0, "confidence": 0.0, "mention_count": 0}
            
            total_sentiment = 0.0
            weighted_sentiment = 0.0
            total_mentions = 0
            
            for post in social_data:
                content = post.get("content", "").lower()
                mentions = post.get("mentions", 1)
                platform = post.get("platform", "unknown")
                
                # Platform-specific weights
                platform_weight = {
                    "twitter": 0.7,
                    "reddit": 0.6,
                    "stocktwits": 0.8,
                    "seeking_alpha": 0.9
                }.get(platform, 0.5)
                
                sentiment_score = self._calculate_text_sentiment(content)
                
                # Weight by mentions and platform
                weight = mentions * platform_weight
                weighted_sentiment += sentiment_score * weight
                total_mentions += mentions
                total_sentiment += sentiment_score
            
            avg_sentiment = weighted_sentiment / total_mentions if total_mentions > 0 else 0.0
            confidence = min(total_mentions / 100.0, 1.0)  # More mentions = higher confidence
            
            return {
                "score": avg_sentiment,
                "confidence": confidence,
                "mention_count": total_mentions,
                "post_count": len(social_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze social sentiment: {str(e)}")
            return {"score": 0.0, "confidence": 0.0, "mention_count": 0}
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score from text using keyword analysis"""
        try:
            if not text:
                return 0.0
            
            # Clean and normalize text
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = text.split()
            
            bullish_score = 0.0
            bearish_score = 0.0
            
            for word in words:
                if word in self.bullish_keywords:
                    bullish_score += self.bullish_keywords[word]
                elif word in self.bearish_keywords:
                    bearish_score += self.bearish_keywords[word]
            
            # Normalize by text length
            word_count = len(words)
            if word_count > 0:
                bullish_score /= word_count
                bearish_score /= word_count
            
            # Calculate net sentiment
            net_sentiment = bullish_score - bearish_score
            
            # Scale to [-1, 1] range
            return max(-1.0, min(1.0, net_sentiment * 10))
            
        except Exception as e:
            logger.error(f"Failed to calculate text sentiment: {str(e)}")
            return 0.0
    
    async def _combine_sentiment_scores(
        self,
        news_sentiment: Dict[str, float],
        social_sentiment: Dict[str, float]
    ) -> Dict[str, Any]:
        """Combine sentiment scores from different sources"""
        try:
            news_score = news_sentiment.get("score", 0.0)
            news_confidence = news_sentiment.get("confidence", 0.0)
            social_score = social_sentiment.get("score", 0.0)
            social_confidence = social_sentiment.get("confidence", 0.0)
            
            # Weight news more heavily than social media
            news_weight = 0.7
            social_weight = 0.3
            
            # Calculate weighted average
            if news_confidence > 0 and social_confidence > 0:
                combined_score = (news_score * news_weight + social_score * social_weight)
                combined_confidence = (news_confidence + social_confidence) / 2
            elif news_confidence > 0:
                combined_score = news_score
                combined_confidence = news_confidence * 0.8  # Reduce confidence when only one source
            elif social_confidence > 0:
                combined_score = social_score
                combined_confidence = social_confidence * 0.6  # Social media less reliable
            else:
                combined_score = 0.0
                combined_confidence = 0.0
            
            return {
                "combined_score": combined_score,
                "combined_confidence": combined_confidence,
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment
            }
            
        except Exception as e:
            logger.error(f"Failed to combine sentiment scores: {str(e)}")
            return {"combined_score": 0.0, "combined_confidence": 0.0}
    
    async def _generate_sentiment_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        sentiment_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """Generate trading signal based on sentiment analysis"""
        try:
            current_price = market_data.get("price", 0)
            if current_price <= 0:
                return None
            
            sentiment_score = sentiment_data.get("combined_score", 0.0)
            sentiment_confidence = sentiment_data.get("combined_confidence", 0.0)
            
            # Require minimum confidence for signal generation
            if sentiment_confidence < 0.3:
                return None
            
            # Determine signal type based on sentiment
            if sentiment_score > 0.3:
                signal_type = SignalType.BUY
                base_confidence = sentiment_score
            elif sentiment_score < -0.3:
                signal_type = SignalType.SELL
                base_confidence = abs(sentiment_score)
            else:
                signal_type = SignalType.HOLD
                base_confidence = 0.3
            
            # Adjust confidence based on sentiment strength and data quality
            final_confidence = base_confidence * sentiment_confidence
            final_confidence = max(0.1, min(0.9, final_confidence))  # Clamp between 0.1 and 0.9
            
            # Determine signal strength
            if final_confidence >= 0.7:
                strength = SignalStrength.STRONG
            elif final_confidence >= 0.5:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Generate reasoning
            reasoning = self._generate_sentiment_reasoning(sentiment_data, signal_type)
            
            # Create signal
            signal = await self.create_signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=final_confidence,
                strength=strength,
                current_price=current_price,
                reasoning=reasoning,
                features={
                    "sentiment_score": sentiment_score,
                    "sentiment_confidence": sentiment_confidence,
                    "news_article_count": sentiment_data.get("news_sentiment", {}).get("article_count", 0),
                    "social_mention_count": sentiment_data.get("social_sentiment", {}).get("mention_count", 0),
                    "news_sentiment": sentiment_data.get("news_sentiment", {}).get("score", 0),
                    "social_sentiment": sentiment_data.get("social_sentiment", {}).get("score", 0)
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate sentiment signal for {symbol}: {str(e)}")
            return None
    
    def _generate_sentiment_reasoning(
        self,
        sentiment_data: Dict[str, Any],
        signal_type: SignalType
    ) -> str:
        """Generate human-readable reasoning for sentiment signal"""
        try:
            news_data = sentiment_data.get("news_sentiment", {})
            social_data = sentiment_data.get("social_sentiment", {})
            
            article_count = news_data.get("article_count", 0)
            mention_count = social_data.get("mention_count", 0)
            news_score = news_data.get("score", 0)
            social_score = social_data.get("score", 0)
            
            reasons = []
            
            # News sentiment reasoning
            if article_count > 0:
                if news_score > 0.3:
                    reasons.append(f"Positive news sentiment from {article_count} articles")
                elif news_score < -0.3:
                    reasons.append(f"Negative news sentiment from {article_count} articles")
                else:
                    reasons.append(f"Neutral news sentiment from {article_count} articles")
            
            # Social sentiment reasoning
            if mention_count > 0:
                if social_score > 0.3:
                    reasons.append(f"Bullish social media sentiment ({mention_count} mentions)")
                elif social_score < -0.3:
                    reasons.append(f"Bearish social media sentiment ({mention_count} mentions)")
                else:
                    reasons.append(f"Mixed social media sentiment ({mention_count} mentions)")
            
            if not reasons:
                reasons.append("Limited sentiment data available")
            
            return f"Sentiment analysis {signal_type.value}: " + "; ".join(reasons)
            
        except Exception as e:
            logger.error(f"Failed to generate sentiment reasoning: {str(e)}")
            return f"Sentiment analysis {signal_type.value} signal" 