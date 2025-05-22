import requests
from textblob import TextBlob
from typing import List, Dict

class NewsSentimentAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://newsapi.org/v2/everything"

    def fetch_news(self, query: str = "stock market", max_articles: int = 10) -> List[Dict]:
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "apiKey": self.api_key
        }
        response = requests.get(self.endpoint, params=params)
        articles = response.json().get("articles", [])
        return articles

    def analyze_sentiment(self, headline: str) -> Dict:
        analysis = TextBlob(headline)
        polarity = analysis.sentiment.polarity
        return {
            "headline": headline,
            "sentiment": "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral",
            "score": polarity
        }

    def fetch_and_analyze(self, topic: str = "TSLA") -> List[Dict]:
        raw_news = self.fetch_news(query=topic)
        return [self.analyze_sentiment(article["title"]) for article in raw_news if "title" in article]
