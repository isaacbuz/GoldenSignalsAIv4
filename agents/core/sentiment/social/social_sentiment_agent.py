import logging
from typing import Any, Dict, List

import requests
from agents.base import BaseAgent
from textblob import TextBlob

logger = logging.getLogger(__name__)

class SocialSentimentAgent(BaseAgent):
    """Agent that analyzes social media sentiment from StockTwits."""

    def __init__(self, stocktwits_base_url: str = "https://api.stocktwits.com/api/2/streams/symbol/"):
        super().__init__(name="SocialSentiment", agent_type="sentiment")
        self.base_url = stocktwits_base_url

    def fetch_messages(self, symbol: str = "TSLA") -> List[str]:
        try:
            url = f"{self.base_url}{symbol}.json"
            response = requests.get(url)
            response.raise_for_status()
            messages = response.json().get("messages", [])
            return [msg.get("body", "") for msg in messages]
        except Exception as e:
            logger.error(f"Failed to fetch StockTwits messages: {e}")
            return []

    def analyze_sentiment(self, text: str) -> Dict:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            return {
                "text": text,
                "sentiment": "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral",
                "score": polarity
            }
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return {"text": text, "sentiment": "neutral", "score": 0.0}

    def hype_score(self, sentiments: List[Dict]) -> float:
        mentions = len(sentiments)
        avg_score = sum(s["score"] for s in sentiments) / mentions if mentions else 0
        return round(avg_score * mentions, 2)  # sentiment * volume

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process social media data to generate trading signals."""
        symbol = data.get("symbol", "TSLA")
        raw_messages = self.fetch_messages(symbol)

        if not raw_messages:
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": "No social media data available"}
            }

        sentiments = [self.analyze_sentiment(msg) for msg in raw_messages if msg]
        avg_sentiment = sum(s["score"] for s in sentiments) / len(sentiments) if sentiments else 0
        hype = self.hype_score(sentiments)

        # Generate trading signal based on sentiment and hype
        if avg_sentiment > 0.2 and hype > 1.0:
            action = "buy"
            confidence = min(avg_sentiment * hype / 2, 1.0)
        elif avg_sentiment < -0.2 and hype > 1.0:
            action = "sell"
            confidence = min(abs(avg_sentiment * hype / 2), 1.0)
        else:
            action = "hold"
            confidence = 0.0

        return {
            "action": action,
            "confidence": confidence,
            "metadata": {
                "symbol": symbol,
                "mentions": len(sentiments),
                "average_sentiment": round(avg_sentiment, 3),
                "hype_score": hype,
                "samples": sentiments[:5]
            }
        }
