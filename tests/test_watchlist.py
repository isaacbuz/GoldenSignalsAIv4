import pytest
from fastapi.testclient import TestClient
from presentation.api.main import app

client = TestClient(app)

def test_watchlist_add_and_get(monkeypatch):
    class DummyManager:
        def add_ticker(self, user_id, ticker, tags):
            pass
        def get_watchlist(self, user_id):
            return [
                {"ticker": "AAPL", "tags": ["tech"]},
                {"ticker": "TSLA", "tags": ["auto", "growth"]}
            ]
    monkeypatch.setattr("application.services.watchlist_manager.WatchlistManager", DummyManager)
    res = client.post("/api/v1/watchlist/add", json={"user_id": "u1", "ticker": "AAPL", "tags": ["tech"]})
    assert res.status_code == 200
    data = res.json()
    assert "watchlist" in data
    assert data["watchlist"][0]["ticker"] == "AAPL"
    res = client.get("/api/v1/watchlist/get/u1")
    assert res.status_code == 200
    data = res.json()
    assert len(data["watchlist"]) == 2
