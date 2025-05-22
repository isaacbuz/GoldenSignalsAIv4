import os
import pytest
from agents.grok.grok_backtest import GrokBacktestCritic
import requests

class DummyResponse:
    def __init__(self, json_data):
        self._json = json_data
    def json(self):
        return self._json

def test_critique(monkeypatch):
    def mock_post(*args, **kwargs):
        return DummyResponse({"suggestions": ["Use a tighter stop loss", "Add RSI filter"]})
    monkeypatch.setattr(requests, "post", mock_post)
    agent = GrokBacktestCritic(api_key="fake-key")
    suggestions = agent.critique("logic", 60, 5.2)
    assert "RSI" in suggestions[1]
