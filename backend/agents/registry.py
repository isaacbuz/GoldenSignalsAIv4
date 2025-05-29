# backend/agents/registry.py

from backend.agents.indicators.rsi_agent import RSIAgent
from backend.agents.indicators.macd_agent import MACDAgent
from backend.agents.indicators.ivrank_agent import IVRankAgent
from backend.agents.ml.forecast_agent import ForecastAgent
from backend.agents.ml.sentiment_agent import SentimentAgent
from backend.agents.news_sentiment_agent import NewsSentimentAgent

class AgentRegistry:
    def __init__(self):
        self.agents = {
            "RSI": RSIAgent(),
            "MACD": MACDAgent(),
            "IVRank": IVRankAgent(),
            "Forecast": ForecastAgent(),
            "Sentiment": SentimentAgent(),
            "NewsSentiment": NewsSentimentAgent()
        }

    def run_all(self, market_data):
        results = {}
        for name, agent in self.agents.items():
            try:
                results[name] = agent.run(market_data)
            except Exception as e:
                results[name] = {"error": str(e), "confidence": 0, "explanation": f"{name} failed"}
        return results
