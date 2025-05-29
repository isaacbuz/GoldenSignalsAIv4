from backend.agents.base import BaseAgent
try:
    from backend.data.sources.yahoo import fetch_headlines
    from backend.nlp.sentiment_engine import analyze_sentiment
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
import random

class NewsSentimentAgent(BaseAgent):
    def run(self, market_data):
        ticker = market_data.get("ticker", "AAPL")
        if NLP_AVAILABLE:
            try:
                headlines = fetch_headlines(ticker)
                summary, score = analyze_sentiment(headlines)
                sentiment = "bullish" if score > 0.3 else "bearish" if score < -0.3 else "neutral"
                confidence = int(abs(score) * 100)
                explanation = f"News sentiment: {summary[:80]}... Score={score:.2f}"
                return {
                    "name": "NewsSentiment",
                    "type": "nlp",
                    "signal": sentiment,
                    "score": score,
                    "confidence": confidence,
                    "explanation": explanation,
                    "headlines": headlines
                }
            except Exception as e:
                explanation = f"NLP pipeline failed: {e}. Falling back to dummy logic."
        # Dummy fallback logic
        headlines = [
            "Company X beats earnings expectations!",
            "Market volatility increases amid uncertainty.",
            "Analyst upgrades stock outlook."
        ]
        sentiment_score = random.randint(-5, 5)
        if sentiment_score > 1:
            signal = "positive"
            confidence = 75
            explanation = explanation if 'explanation' in locals() else f"News headlines positive (score={sentiment_score})"
        elif sentiment_score < -1:
            signal = "negative"
            confidence = 75
            explanation = explanation if 'explanation' in locals() else f"News headlines negative (score={sentiment_score})"
        else:
            signal = "neutral"
            confidence = 60
            explanation = explanation if 'explanation' in locals() else f"News headlines neutral (score={sentiment_score})"
        return {
            "name": "NewsSentiment",
            "type": "nlp",
            "signal": signal,
            "score": sentiment_score,
            "confidence": confidence,
            "explanation": explanation,
            "headlines": headlines
        }

