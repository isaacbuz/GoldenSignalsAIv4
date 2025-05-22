import requests
from textblob import TextBlob
from typing import List, Dict

class SocialSentimentAgent:
    def __init__(self, stocktwits_base_url: str = "https://api.stocktwits.com/api/2/streams/symbol/"):
        self.base_url = stocktwits_base_url

    def fetch_messages(self, symbol: str = "TSLA") -> List[str]:
        url = f"{self.base_url}{symbol}.json"
        response = requests.get(url)
        messages = response.json().get("messages", [])
        return [msg.get("body", "") for msg in messages]

    def analyze_sentiment(self, text: str) -> Dict:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        return {
            "text": text,
            "sentiment": "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral",
            "score": polarity
        }

    def hype_score(self, sentiments: List[Dict]) -> float:
        mentions = len(sentiments)
        avg_score = sum(s["score"] for s in sentiments) / mentions if mentions else 0
        return round(avg_score * mentions, 2)  # sentiment * volume

    def fetch_social_sentiment(self, symbol: str = "TSLA") -> Dict:
        raw_messages = self.fetch_messages(symbol)
        sentiments = [self.analyze_sentiment(msg) for msg in raw_messages if msg]
        return {
            "symbol": symbol,
            "mentions": len(sentiments),
            "average_sentiment": round(sum(s["score"] for s in sentiments) / len(sentiments), 3) if sentiments else 0,
            "hype_score": self.hype_score(sentiments),
            "samples": sentiments[:5]
        }
