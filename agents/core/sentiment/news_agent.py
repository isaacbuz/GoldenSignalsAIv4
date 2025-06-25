"""
News Agent - Real-time news analysis and event detection for trading signals.
Analyzes news headlines, sentiment, velocity, and market-moving events.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from src.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class NewsAgent(BaseAgent):
    """Agent that analyzes news flow and generates sentiment-based trading signals."""
    
    def __init__(
        self,
        name: str = "News",
        sentiment_threshold: float = 0.6,
        news_velocity_period: int = 60,  # minutes
        impact_keywords_weight: float = 2.0,
        source_credibility_weights: Optional[Dict[str, float]] = None,
        event_keywords: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize News agent.
        
        Args:
            name: Agent name
            sentiment_threshold: Threshold for significant sentiment
            news_velocity_period: Period for measuring news velocity
            impact_keywords_weight: Weight multiplier for high-impact keywords
            source_credibility_weights: Weight by news source credibility
            event_keywords: Keywords for different event types
        """
        super().__init__(name=name, agent_type="sentiment")
        self.sentiment_threshold = sentiment_threshold
        self.news_velocity_period = news_velocity_period
        self.impact_keywords_weight = impact_keywords_weight
        
        # Default source credibility weights
        self.source_weights = source_credibility_weights or {
            'reuters': 1.0,
            'bloomberg': 1.0,
            'wsj': 0.95,
            'cnbc': 0.8,
            'marketwatch': 0.7,
            'seekingalpha': 0.6,
            'yahoo': 0.5,
            'twitter': 0.3
        }
        
        # Event detection keywords
        self.event_keywords = event_keywords or {
            'earnings': ['earnings', 'eps', 'quarterly results', 'guidance', 'revenue'],
            'merger': ['merger', 'acquisition', 'takeover', 'buyout', 'deal'],
            'regulatory': ['fda', 'sec', 'investigation', 'lawsuit', 'compliance'],
            'product': ['product launch', 'new product', 'recall', 'drug approval'],
            'management': ['ceo', 'cfo', 'resignation', 'appointment', 'leadership'],
            'financial': ['bankruptcy', 'debt', 'credit', 'loan', 'bond'],
            'market': ['upgrade', 'downgrade', 'target price', 'rating', 'analyst']
        }
        
        # Positive/negative keywords for sentiment
        self.positive_keywords = [
            'beat', 'exceed', 'strong', 'growth', 'profit', 'gain', 'rise', 'surge',
            'outperform', 'bullish', 'positive', 'upgrade', 'approval', 'success'
        ]
        
        self.negative_keywords = [
            'miss', 'decline', 'fall', 'drop', 'loss', 'weak', 'concern', 'risk',
            'underperform', 'bearish', 'negative', 'downgrade', 'rejection', 'failure'
        ]
        
        # High-impact keywords that significantly move markets
        self.high_impact_keywords = [
            'bankruptcy', 'acquisition', 'merger', 'fda approval', 'clinical trial',
            'earnings beat', 'earnings miss', 'guidance', 'dividend', 'split'
        ]
    
    def calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score for a piece of text."""
        try:
            if not text:
                return 0.0
            
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            positive_count = sum(1 for word in words if word in self.positive_keywords)
            negative_count = sum(1 for word in words if word in self.negative_keywords)
            
            total_sentiment_words = positive_count + negative_count
            
            if total_sentiment_words == 0:
                return 0.0
            
            # Score between -1 (very negative) and 1 (very positive)
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Sentiment score calculation failed: {str(e)}")
            return 0.0
    
    def detect_events(self, text: str) -> Dict[str, Any]:
        """Detect specific events mentioned in news text."""
        try:
            detected_events = []
            text_lower = text.lower()
            
            for event_type, keywords in self.event_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        detected_events.append({
                            'type': event_type,
                            'keyword': keyword,
                            'confidence': 0.8 if len(keyword.split()) > 1 else 0.6
                        })
            
            # Check for high-impact events
            high_impact_detected = []
            for keyword in self.high_impact_keywords:
                if keyword in text_lower:
                    high_impact_detected.append(keyword)
            
            return {
                'events': detected_events,
                'event_types': list(set([e['type'] for e in detected_events])),
                'high_impact_events': high_impact_detected,
                'event_count': len(detected_events)
            }
            
        except Exception as e:
            logger.error(f"Event detection failed: {str(e)}")
            return {'events': [], 'event_types': [], 'high_impact_events': [], 'event_count': 0}
    
    def calculate_news_velocity(self, news_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate the velocity of news flow."""
        try:
            if not news_items:
                return {'velocity': 0.0, 'acceleration': 0.0}
            
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(minutes=self.news_velocity_period)
            
            # Count recent news
            recent_news = [
                item for item in news_items 
                if 'timestamp' in item and 
                datetime.fromisoformat(item['timestamp']) >= cutoff_time
            ]
            
            velocity = len(recent_news) / (self.news_velocity_period / 60)  # News per hour
            
            # Calculate acceleration (change in velocity)
            half_period = self.news_velocity_period // 2
            half_cutoff = current_time - timedelta(minutes=half_period)
            
            very_recent = [
                item for item in recent_news 
                if datetime.fromisoformat(item['timestamp']) >= half_cutoff
            ]
            
            recent_velocity = len(very_recent) / (half_period / 60)
            earlier_velocity = (len(recent_news) - len(very_recent)) / (half_period / 60)
            
            acceleration = recent_velocity - earlier_velocity
            
            return {
                'velocity': velocity,
                'acceleration': acceleration,
                'recent_count': len(recent_news),
                'very_recent_count': len(very_recent)
            }
            
        except Exception as e:
            logger.error(f"News velocity calculation failed: {str(e)}")
            return {'velocity': 0.0, 'acceleration': 0.0, 'recent_count': 0, 'very_recent_count': 0}
    
    def analyze_source_credibility(self, news_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze news source credibility and coverage."""
        try:
            if not news_items:
                return {'avg_credibility': 0.5, 'source_diversity': 0.0}
            
            sources = [item.get('source', 'unknown').lower() for item in news_items]
            source_counts = Counter(sources)
            
            # Calculate weighted average credibility
            total_weight = 0
            weighted_credibility = 0
            
            for source, count in source_counts.items():
                weight = self.source_weights.get(source, 0.5)  # Default credibility
                total_weight += count * weight
                weighted_credibility += count * weight * weight
            
            avg_credibility = weighted_credibility / total_weight if total_weight > 0 else 0.5
            
            # Source diversity (number of unique sources)
            source_diversity = len(source_counts) / max(len(news_items), 1)
            
            return {
                'avg_credibility': avg_credibility,
                'source_diversity': source_diversity,
                'source_counts': dict(source_counts),
                'unique_sources': len(source_counts)
            }
            
        except Exception as e:
            logger.error(f"Source credibility analysis failed: {str(e)}")
            return {'avg_credibility': 0.5, 'source_diversity': 0.0}
    
    def calculate_market_impact_score(self, news_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate potential market impact of news."""
        try:
            if not news_items:
                return {'impact_score': 0.0, 'sentiment_momentum': 0.0}
            
            total_impact = 0.0
            sentiment_scores = []
            
            for item in news_items:
                text = item.get('headline', '') + ' ' + item.get('content', '')
                
                # Base sentiment score
                sentiment = self.calculate_sentiment_score(text)
                sentiment_scores.append(sentiment)
                
                # Event detection and impact
                events = self.detect_events(text)
                
                # Calculate impact multiplier
                impact_multiplier = 1.0
                
                # High-impact events
                if events['high_impact_events']:
                    impact_multiplier *= self.impact_keywords_weight
                
                # Multiple events increase impact
                if events['event_count'] > 1:
                    impact_multiplier *= 1.5
                
                # Source credibility
                source = item.get('source', 'unknown').lower()
                credibility = self.source_weights.get(source, 0.5)
                impact_multiplier *= credibility
                
                # Time decay (recent news has more impact)
                if 'timestamp' in item:
                    try:
                        news_time = datetime.fromisoformat(item['timestamp'])
                        hours_ago = (datetime.now() - news_time).total_seconds() / 3600
                        time_decay = max(0.1, 1.0 - (hours_ago / 24))  # Decay over 24 hours
                        impact_multiplier *= time_decay
                    except:
                        pass
                
                item_impact = abs(sentiment) * impact_multiplier
                total_impact += item_impact
            
            # Calculate sentiment momentum (trend in sentiment)
            sentiment_momentum = 0.0
            if len(sentiment_scores) >= 3:
                recent_sentiment = np.mean(sentiment_scores[-3:])
                earlier_sentiment = np.mean(sentiment_scores[:-3]) if len(sentiment_scores) > 3 else 0
                sentiment_momentum = recent_sentiment - earlier_sentiment
            
            return {
                'impact_score': total_impact,
                'sentiment_momentum': sentiment_momentum,
                'avg_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0.0,
                'sentiment_volatility': np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Market impact calculation failed: {str(e)}")
            return {'impact_score': 0.0, 'sentiment_momentum': 0.0}
    
    def generate_news_signals(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on news analysis."""
        try:
            news_items = news_data.get('news_items', [])
            symbol = news_data.get('symbol', 'UNKNOWN')
            
            if not news_items:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'signal_type': 'no_news',
                    'reasoning': 'No news data available'
                }
            
            # Analyze all components
            velocity_data = self.calculate_news_velocity(news_items)
            credibility_data = self.analyze_source_credibility(news_items)
            impact_data = self.calculate_market_impact_score(news_items)
            
            # Generate primary signal
            action = "hold"
            confidence = 0.0
            signal_type = "neutral"
            reasoning = []
            
            # Sentiment-based signals
            avg_sentiment = impact_data['avg_sentiment']
            
            if abs(avg_sentiment) >= self.sentiment_threshold:
                if avg_sentiment > 0:
                    action = "buy"
                    signal_type = "positive_sentiment"
                    reasoning.append(f"Strong positive sentiment: {avg_sentiment:.2f}")
                else:
                    action = "sell"
                    signal_type = "negative_sentiment"
                    reasoning.append(f"Strong negative sentiment: {avg_sentiment:.2f}")
                
                confidence += min(abs(avg_sentiment), 1.0) * 0.5
            
            # Impact score influence
            if impact_data['impact_score'] > 2.0:
                confidence += 0.3
                reasoning.append(f"High market impact score: {impact_data['impact_score']:.1f}")
            
            # News velocity influence
            if velocity_data['velocity'] > 5.0:  # More than 5 news per hour
                confidence += 0.2
                reasoning.append(f"High news velocity: {velocity_data['velocity']:.1f} per hour")
                
                # Acceleration can indicate building momentum
                if velocity_data['acceleration'] > 2.0:
                    confidence += 0.1
                    reasoning.append("Accelerating news flow")
            
            # Sentiment momentum
            if abs(impact_data['sentiment_momentum']) > 0.3:
                if impact_data['sentiment_momentum'] > 0:
                    if action == "hold":
                        action = "buy"
                    confidence += 0.2
                    reasoning.append("Improving sentiment trend")
                else:
                    if action == "hold":
                        action = "sell"
                    confidence += 0.2
                    reasoning.append("Deteriorating sentiment trend")
            
            # Source credibility boost
            if credibility_data['avg_credibility'] > 0.8:
                confidence *= 1.2
                reasoning.append("High credibility sources")
            elif credibility_data['avg_credibility'] < 0.5:
                confidence *= 0.8
                reasoning.append("Lower credibility sources")
            
            # Source diversity boost
            if credibility_data['source_diversity'] > 0.5:
                confidence *= 1.1
                reasoning.append("Diverse source coverage")
            
            return {
                'action': action,
                'confidence': min(1.0, confidence),
                'signal_type': signal_type,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"News signal generation failed: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'signal_type': 'error',
                'reasoning': [str(e)]
            }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process news data and generate sentiment-based signals."""
        try:
            if "news_data" not in data:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "No news data provided"}
                }
            
            news_data = data["news_data"]
            
            # Generate signals
            signal_data = self.generate_news_signals(news_data)
            
            # Get comprehensive analysis
            news_items = news_data.get('news_items', [])
            velocity_data = self.calculate_news_velocity(news_items)
            credibility_data = self.analyze_source_credibility(news_items)
            impact_data = self.calculate_market_impact_score(news_items)
            
            # Analyze events in all news
            all_events = []
            for item in news_items:
                text = item.get('headline', '') + ' ' + item.get('content', '')
                events = self.detect_events(text)
                all_events.extend(events['events'])
            
            return {
                "action": signal_data['action'],
                "confidence": signal_data['confidence'],
                "metadata": {
                    "signal_type": signal_data['signal_type'],
                    "reasoning": signal_data['reasoning'],
                    "news_velocity": velocity_data,
                    "source_credibility": credibility_data,
                    "market_impact": impact_data,
                    "detected_events": all_events,
                    "news_count": len(news_items)
                }
            }
            
        except Exception as e:
            logger.error(f"News signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 