import pytest
import requests

# Adjust these as needed for your local or test server setup
API_BASE_URL = "http://localhost:8000"

def test_dashboard_endpoint():
    symbol = "AAPL"
    url = f"{API_BASE_URL}/dashboard/{symbol}"
    response = requests.get(url)
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data and data["symbol"] == symbol
    assert "price" in data
    assert "trend" in data
    assert "options_data" in data
    assert all(k in data["options_data"] for k in ["iv", "delta", "gamma", "theta"])

def test_prediction_endpoint():
    url = f"{API_BASE_URL}/predict"
    payload = {"symbol": "AAPL", "features": [1,2,3,4]}  # Adjust features as needed
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data

def test_health_endpoint():
    url = f"{API_BASE_URL}/health"
    response = requests.get(url)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
