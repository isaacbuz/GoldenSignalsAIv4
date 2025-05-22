import requests
import pytest

BASE_URL = "http://localhost:8000"

@pytest.mark.parametrize("endpoint,payload,expected_key", [
    ("/finbert_sentiment/analyze", {"texts": ["Stocks are up.", "Bad news for the market."]}, "average_score"),
    ("/lstm_forecast/predict", {"series": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]}, "prediction"),
    ("/ml_classifier/predict", {"features": [[0.5, 0.2, 0.1]]}, "signal"),
    ("/rsi_macd/predict", {"ohlcv": [{"close": 100}, {"close": 102}, {"close": 101}, {"close": 105}, {"close": 110}]}, "signal"),
])
def test_api_endpoints(endpoint, payload, expected_key):
    resp = requests.post(BASE_URL + endpoint, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert expected_key in data
