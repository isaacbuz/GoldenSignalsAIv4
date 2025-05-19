import pytest
from fastapi.testclient import TestClient
from presentation.api.main import app

client = TestClient(app)

def test_dashboard_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in ["ok", "healthy"]

def test_dashboard_predict():
    # This test assumes authentication is relaxed for testing
    resp = client.post("/predict", json={"symbol": "AAPL"})
    assert resp.status_code == 200
    assert "Prediction successful" in resp.json().get("status", "")
