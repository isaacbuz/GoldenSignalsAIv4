from backend.agents.base import BaseSignalAgent
from transformers import pipeline

class NewsSentimentAgent(BaseSignalAgent):
    """
    Uses HuggingFace Transformers to analyze sentiment of news headlines for a symbol.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def run(self, news_headlines: list) -> dict:
        if not news_headlines:
            return {"agent": "NewsSentimentAgent", "signal": "neutral", "confidence": 0, "explanation": "No news provided."}
        results = self.sentiment_pipeline(news_headlines)
        pos = sum(1 for r in results if r['label'] == 'POSITIVE')
        neg = sum(1 for r in results if r['label'] == 'NEGATIVE')
        neu = len(results) - pos - neg
        if pos > neg:
            signal = "positive"
        elif neg > pos:
            signal = "negative"
        else:
            signal = "neutral"
        explanation = f"{pos} positive, {neg} negative, {neu} neutral headlines."
        return {"agent": "NewsSentimentAgent", "signal": signal, "confidence": 70, "explanation": explanation}
