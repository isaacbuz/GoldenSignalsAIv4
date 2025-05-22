import os
import pytest
from agents.grok.grok_strategy import GrokStrategyAgent
import requests

class DummyResponse:
    def __init__(self, json_data):
        self._json = json_data
    def json(self):
        return self._json

def test_generate_logic(monkeypatch):
    def mock_post(*args, **kwargs):
        return DummyResponse({"logic": "buy if EMA9 > price"})
    monkeypatch.setattr(requests, "post", mock_post)
    agent = GrokStrategyAgent(api_key="fake-key")
    logic = agent.generate_logic("AAPL")
    assert logic == "buy if EMA9 > price"
