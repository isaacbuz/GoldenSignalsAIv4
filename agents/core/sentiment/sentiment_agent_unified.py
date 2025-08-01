"""
Unified Sentiment Analysis Agent
Analyzes market sentiment from multiple sources using MCP tools
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from agents.unified_base_agent import SignalStrength, UnifiedBaseAgent

logger = logging.getLogger(__name__)


class SentimentAgent(UnifiedBaseAgent):
    """
    Unified sentiment agent that analyzes:
    - News sentiment (via MCP tools)
    - Social media sentiment
    - Market fear/greed indicators
    - Options sentiment (put/call ratio)
    - Analyst ratings consensus
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="SentimentAgent",
            weight=0.9,  # Lower weight due to sentiment noise
            config=config
        )

        # Configuration
        self.lookback_hours = self.config.get("lookback_hours", 24)
        self.min_sources = self.config.get("min_sources", 3)
        self.social_weight = self.config.get("social_weight", 0.3)
        self.news_weight = self.config.get("news_weight", 0.4)
        self.market_weight = self.config.get("market_weight", 0.3)

    def get_required_data_fields(self) -> List[str]:
        """Required fields for sentiment analysis"""
        return ["symbol", "current_price"]

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market sentiment from multiple sources
        """
        try:
            symbol = market_data["symbol"]

            # Get sentiment from MCP tools
            sentiment_data = await self.get_sentiment_analysis(symbol)

            # Get additional market data for context
            options_flow = await self.get_options_flow(symbol)

            # Initialize sentiment components
            sentiment_scores = []
            confidence_factors = []
            reasoning_parts = []

            # 1. News Sentiment (from MCP)
            news_sentiment = sentiment_data.get("news_sentiment", 0)
            news_confidence = sentiment_data.get("news_confidence", 0.5)

            if news_confidence > 0.3:
                sentiment_scores.append((news_sentiment, self.news_weight))
                confidence_factors.append(news_confidence)

                if abs(news_sentiment) > 0.3:
                    sentiment_type = "positive" if news_sentiment > 0 else "negative"
                    reasoning_parts.append(f"Strong {sentiment_type} news sentiment")

            # 2. Social Media Sentiment (from MCP)
            social_sentiment = sentiment_data.get("social_sentiment", 0)
            social_volume = sentiment_data.get("social_volume", 0)

            if social_volume > 100:  # Minimum volume threshold
                sentiment_scores.append((social_sentiment, self.social_weight))
                confidence_factors.append(min(0.7, social_volume / 1000))

                if abs(social_sentiment) > 0.4:
                    sentiment_type = "bullish" if social_sentiment > 0 else "bearish"
                    reasoning_parts.append(f"High {sentiment_type} social media activity")

            # 3. Options Sentiment
            if options_flow:
                put_call_ratio = options_flow.get("put_call_ratio", 1.0)
                options_sentiment = self._calculate_options_sentiment(put_call_ratio)

                sentiment_scores.append((options_sentiment, self.market_weight))
                confidence_factors.append(0.8)  # Options data is generally reliable

                if put_call_ratio > 1.3:
                    reasoning_parts.append("High put/call ratio indicates bearish sentiment")
                elif put_call_ratio < 0.7:
                    reasoning_parts.append("Low put/call ratio indicates bullish sentiment")

            # 4. Market Fear/Greed Indicators
            market_indicators = self._calculate_market_indicators(market_data)
            if market_indicators:
                fear_greed_score = market_indicators["fear_greed"]
                sentiment_scores.append((fear_greed_score, self.market_weight * 0.5))
                confidence_factors.append(0.6)

                if fear_greed_score < -0.5:
                    reasoning_parts.append("Market showing extreme fear")
                elif fear_greed_score > 0.5:
                    reasoning_parts.append("Market showing extreme greed")

            # Calculate weighted sentiment score
            if sentiment_scores:
                total_weight = sum(weight for _, weight in sentiment_scores)
                weighted_sentiment = sum(score * weight for score, weight in sentiment_scores) / total_weight

                # Normalize to [-1, 1]
                weighted_sentiment = max(-1.0, min(1.0, weighted_sentiment))
            else:
                weighted_sentiment = 0.0
                reasoning_parts.append("Insufficient sentiment data available")

            # Calculate confidence
            if confidence_factors:
                confidence = np.mean(confidence_factors)
                # Adjust confidence based on number of sources
                source_multiplier = min(1.0, len(sentiment_scores) / self.min_sources)
                confidence *= source_multiplier
            else:
                confidence = 0.2

            # Build reasoning
            if not reasoning_parts:
                if weighted_sentiment > 0.1:
                    reasoning_parts.append("Mildly positive market sentiment")
                elif weighted_sentiment < -0.1:
                    reasoning_parts.append("Mildly negative market sentiment")
                else:
                    reasoning_parts.append("Neutral market sentiment")

            reasoning = "; ".join(reasoning_parts)

            # Additional data for transparency
            analysis_data = {
                "news_sentiment": round(news_sentiment, 3),
                "social_sentiment": round(social_sentiment, 3),
                "weighted_sentiment": round(weighted_sentiment, 3),
                "sources_analyzed": len(sentiment_scores),
                "put_call_ratio": options_flow.get("put_call_ratio") if options_flow else None,
                "sentiment_components": [
                    {"source": "news", "score": news_sentiment, "weight": self.news_weight},
                    {"source": "social", "score": social_sentiment, "weight": self.social_weight}
                ]
            }

            return {
                "signal": weighted_sentiment,
                "confidence": confidence,
                "reasoning": reasoning,
                "sentiment_score": weighted_sentiment,
                "data": analysis_data
            }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {
                "signal": 0,
                "confidence": 0.1,
                "reasoning": f"Sentiment analysis error: {str(e)}"
            }

    def _calculate_options_sentiment(self, put_call_ratio: float) -> float:
        """
        Convert put/call ratio to sentiment score
        High put/call ratio (>1.2) is bearish
        Low put/call ratio (<0.8) is bullish
        """
        if put_call_ratio > 2.0:
            return -0.9  # Extremely bearish
        elif put_call_ratio > 1.5:
            return -0.6  # Very bearish
        elif put_call_ratio > 1.2:
            return -0.3  # Bearish
        elif put_call_ratio > 0.8:
            return 0.0   # Neutral
        elif put_call_ratio > 0.6:
            return 0.3   # Bullish
        elif put_call_ratio > 0.4:
            return 0.6   # Very bullish
        else:
            return 0.9   # Extremely bullish

    def _calculate_market_indicators(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Calculate market-wide fear/greed indicators
        This is simplified - in production, would use actual VIX, market breadth, etc.
        """
        try:
            # Simplified fear/greed calculation based on price momentum
            historical_data = market_data.get("historical_data", [])
            if len(historical_data) < 20:
                return None

            # Recent price momentum
            recent_prices = [d["close"] for d in historical_data[-20:]]
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            # Volatility (simplified)
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns)

            # Convert to fear/greed score
            # High volatility + negative returns = fear
            # Low volatility + positive returns = greed
            if price_change < -0.05 and volatility > 0.02:
                fear_greed = -0.7  # Fear
            elif price_change < -0.02:
                fear_greed = -0.4  # Mild fear
            elif price_change > 0.05 and volatility < 0.01:
                fear_greed = 0.7   # Greed
            elif price_change > 0.02:
                fear_greed = 0.4   # Mild greed
            else:
                fear_greed = 0.0   # Neutral

            return {
                "fear_greed": fear_greed,
                "price_momentum": price_change,
                "volatility": volatility
            }

        except Exception as e:
            logger.error(f"Market indicators calculation error: {e}")
            return None

    async def get_enhanced_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get enhanced sentiment analysis with additional sources
        This method can be called directly for more detailed analysis
        """
        try:
            # Get base sentiment
            base_sentiment = await self.get_sentiment_analysis(symbol)

            # Get technical indicators for sentiment context
            technicals = await self.get_technical_indicators(
                symbol,
                ["RSI", "MACD"]
            )

            # Enhance sentiment with technical context
            rsi = technicals.get("RSI", 50)

            # Adjust sentiment based on oversold/overbought conditions
            if rsi < 30:
                # Oversold - potential bullish reversal
                base_sentiment["technical_bias"] = 0.3
                base_sentiment["technical_context"] = "Oversold conditions may support reversal"
            elif rsi > 70:
                # Overbought - potential bearish reversal
                base_sentiment["technical_bias"] = -0.3
                base_sentiment["technical_context"] = "Overbought conditions may limit upside"
            else:
                base_sentiment["technical_bias"] = 0.0
                base_sentiment["technical_context"] = "Normal technical conditions"

            return base_sentiment

        except Exception as e:
            logger.error(f"Enhanced sentiment error: {e}")
            return {"error": str(e)}


# Export for compatibility
__all__ = ['SentimentAgent']
