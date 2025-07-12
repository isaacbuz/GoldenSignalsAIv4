import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_get_signals():
    response = client.get("/api/v1/signals")
    assert response.status_code == 200

def test_create_signal():
    signal_data = {"symbol": "AAPL", "signal_type": "BUY"}
    response = client.post("/api/v1/signals", json=signal_data)
    assert response.status_code in [200, 201]