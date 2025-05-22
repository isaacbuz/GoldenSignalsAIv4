import os
import pytest
from agents.grok.grok_sentiment import GrokSentimentAgent
import requests

class DummyResponse:
    def __init__(self, json_data):
        self._json = json_data
    def json(self):
        return self._json

def test_get_sentiment_score(monkeypatch):
    def mock_post(*args, **kwargs):
        return DummyResponse({"sentimentScore": 87})
    monkeypatch.setattr(requests, "post", mock_post)
    agent = GrokSentimentAgent(api_key="fake-key")
    score = agent.get_sentiment_score("AAPL")
    assert score == 87
