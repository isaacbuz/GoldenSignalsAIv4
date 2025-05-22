import pytest
from fastapi.testclient import TestClient
from presentation.api.main import app

client = TestClient(app)

def test_news_sentiment_success(monkeypatch):
    # Patch NewsSentimentAgent to avoid real API calls
    class DummyAgent:
        def fetch_and_analyze(self, topic):
            return [
                {"headline": "TSLA rallies on earnings", "sentiment": "positive", "score": 0.8},
                {"headline": "TSLA faces supply chain issues", "sentiment": "negative", "score": -0.5}
            ]
    monkeypatch.setattr("agents.sentiment.news_agent.NewsSentimentAgent", DummyAgent)
    res = client.post("/api/v1/news_agent/analyze", json={"topic": "TSLA", "max_articles": 2})
    assert res.status_code == 200
    data = res.json()
    assert "headlines" in data
    assert len(data["headlines"]) == 2
    assert data["headlines"][0]["sentiment"] == "positive"

def test_news_sentiment_error(monkeypatch):
    class DummyAgent:
        def fetch_and_analyze(self, topic):
            raise Exception("API error")
    monkeypatch.setattr("agents.sentiment.news_agent.NewsSentimentAgent", DummyAgent)
    res = client.post("/api/v1/news_agent/analyze", json={"topic": "TSLA", "max_articles": 2})
    assert res.status_code == 500
