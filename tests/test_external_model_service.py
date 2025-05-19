"""
Unit tests for ExternalModelService (agentic foundation model abstraction)
Covers async, caching, fallback, and ensemble logic.
"""
import asyncio
import pytest
from integration.external_model_service import ExternalModelService

@pytest.mark.asyncio
async def test_analyze_sentiment_caching(monkeypatch):
    service = ExternalModelService()
    text = "Market is bullish!"
    # Patch the adapter to count calls
    call_count = {"count": 0}
    def fake_analyze_sentiment(self, text):
        call_count["count"] += 1
        return {"sentiment": "positive", "confidence": 0.99}
    monkeypatch.setattr(service.adapters["claude"], "analyze_sentiment", fake_analyze_sentiment.__get__(service.adapters["claude"]))
    # First call: should hit adapter
    result1 = await service.analyze_sentiment(text, provider="claude")
    # Second call: should hit cache
    result2 = await service.analyze_sentiment(text, provider="claude")
    assert result1 == result2
    assert call_count["count"] == 1

@pytest.mark.asyncio
async def test_fallback_on_failure(monkeypatch):
    service = ExternalModelService()
    text = "Fed raises rates."
    # Patch primary adapter to fail
    def fail_sentiment(self, text):
        raise Exception("API Down")
    monkeypatch.setattr(service.adapters["claude"], "analyze_sentiment", fail_sentiment.__get__(service.adapters["claude"]))
    # Patch fallback to succeed
    def ok_sentiment(self, text):
        return {"sentiment": "neutral", "confidence": 0.8}
    monkeypatch.setattr(service.adapters["llama"], "analyze_sentiment", ok_sentiment.__get__(service.adapters["llama"]))
    result = await service.analyze_sentiment(text, provider="claude")
    assert result["sentiment"] == "neutral"
    assert result["confidence"] == 0.8

@pytest.mark.asyncio
async def test_ensemble_sentiment_parallel(monkeypatch):
    service = ExternalModelService()
    text = "AAPL earnings report."
    # Patch all adapters to return different sentiments
    monkeypatch.setattr(service.adapters["claude"], "analyze_sentiment", lambda self, t: {"sentiment": "positive", "confidence": 0.9})
    monkeypatch.setattr(service.adapters["llama"], "analyze_sentiment", lambda self, t: {"sentiment": "negative", "confidence": 0.7})
    monkeypatch.setattr(service.adapters["titan"], "analyze_sentiment", lambda self, t: {"sentiment": "positive", "confidence": 0.8})
    monkeypatch.setattr(service.adapters["cohere"], "analyze_sentiment", lambda self, t: {"sentiment": "neutral", "confidence": 0.6})
    monkeypatch.setattr(service.adapters["grok"], "analyze_sentiment", lambda self, t: {"sentiment": "positive", "confidence": 0.95})
    result = await service.ensemble_sentiment(text)
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.8
    assert "details" in result
