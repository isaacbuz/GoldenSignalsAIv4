from typing import Dict, Any
# agents/sentiment/social_media.py
# Purpose: Implements a SocialMediaSentimentAgent that analyzes social media sentiment
# to generate trading signals, useful for options trading during sentiment-driven volatility.

import logging

import pandas as pd
# Simple sentiment analysis using word lists
POSITIVE_WORDS = {'good', 'great', 'excellent', 'bullish', 'buy', 'positive', 'strong', 'gain', 'rise', 'up'}
NEGATIVE_WORDS = {'bad', 'poor', 'terrible', 'bearish', 'sell', 'negative', 'weak', 'loss', 'fall', 'down'}

from ..base_agent import BaseAgent

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class SocialMediaSentimentAgent(BaseAgent):
    """Agent that analyzes social media sentiment for trading signals."""

    def __init__(self, sentiment_threshold: float = 0.3):
        """Initialize the SocialMediaSentimentAgent.

        Args:
            sentiment_threshold (float): Threshold for significant sentiment score.
        """
        self.sentiment_threshold = sentiment_threshold
        # Sentiment analyzer removed, using custom word-based approach
        logger.info(
            {
                "message": f"SocialMediaSentimentAgent initialized with sentiment_threshold={sentiment_threshold}"
            }
        )

    def process(self, data: Dict) -> Dict:
        """Process social media data to analyze sentiment.

        Args:
            data (Dict): Market observation with 'social_media'.

        Returns:
            Dict: Decision with 'action', 'confidence', and 'metadata'.
        """
        logger.info({"message": "Processing data for SocialMediaSentimentAgent"})
        try:
            social_media = data["social_media"]
            if not social_media:
                logger.warning({"message": "No social media data available"})
                return {"action": "hold", "confidence": 0.0, "metadata": {}}

            # Analyze sentiment
            # Simple word-based sentiment analysis
            sentiment_scores = []
            for post in social_media:
                text = post.get('text', '').lower()
                if not text:
                    continue
                
                # Count positive and negative words
                positive_count = sum(1 for word in text.split() if word in POSITIVE_WORDS)
                negative_count = sum(1 for word in text.split() if word in NEGATIVE_WORDS)
                
                # Calculate sentiment score
                if positive_count > negative_count:
                    sentiment_scores.append(positive_count / (positive_count + negative_count))
                elif negative_count > positive_count:
                    sentiment_scores.append(-negative_count / (positive_count + negative_count))
                else:
                    sentiment_scores.append(0.0)
            
            avg_sentiment = (
                sum(sentiment_scores) / len(sentiment_scores)
                if sentiment_scores
                else 0.0
            )

            # Generate trading signal
            if avg_sentiment > self.sentiment_threshold:
                action = "buy"  # Positive sentiment
                confidence = avg_sentiment
            elif avg_sentiment < -self.sentiment_threshold:
                action = "sell"  # Negative sentiment
                confidence = abs(avg_sentiment)
            else:
                action = "hold"
                confidence = 0.0

            decision = {
                "action": action,
                "confidence": min(confidence, 1.0),
                "metadata": {"avg_sentiment": avg_sentiment},
            }
            logger.info({"message": f"SocialMediaSentimentAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error(
                {"message": f"SocialMediaSentimentAgent processing failed: {str(e)}"}
            )
            return {"action": "hold", "confidence": 0.0, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        """Adapt the agent to new social media data (placeholder for learning).

        Args:
            new_data (pd.DataFrame): New social media data.
        """
        logger.info({"message": "SocialMediaSentimentAgent adapting to new data"})
        try:
            # Placeholder: Adjust threshold based on sentiment trends
            pass
        except Exception as e:
            logger.error(
                {"message": f"SocialMediaSentimentAgent adaptation failed: {str(e)}"}
            )

    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Process and potentially modify a trading signal.
        
        Args:
            signal (Dict[str, Any]): Input trading signal
            
        Returns:
            Dict[str, Any]: Potentially modified trading signal
        """
        # Default implementation: return signal as-is
        return signals