import requests
import pytest

BASE_URL = "http://localhost:8000"

def test_finbert_sentiment_empty():
    resp = requests.post(BASE_URL + "/finbert_sentiment/analyze", json={"texts": []})
    assert resp.status_code == 200
    assert resp.json()["average_score"] == 0.0
    assert "error" in str(resp.json()["raw_results"][0])

def test_lstm_forecast_short_series():
    resp = requests.post(BASE_URL + "/lstm_forecast/predict", json={"series": [1.0]*10})
    assert resp.status_code == 200
    assert "error" in str(resp.json()["prediction"])

def test_ml_classifier_empty_features():
    resp = requests.post(BASE_URL + "/ml_classifier/predict", json={"features": []})
    assert resp.status_code == 200
    assert "error" in str(resp.json()["signal"])

def test_rsi_macd_short_ohlcv():
    resp = requests.post(BASE_URL + "/rsi_macd/predict", json={"ohlcv": [{"close": 100}] * 5})
    assert resp.status_code == 200
    assert "error" in str(resp.json()["signal"])
