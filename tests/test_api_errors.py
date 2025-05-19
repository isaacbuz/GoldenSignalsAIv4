import requests

def test_invalid_symbol():
    url = "http://localhost:8000/dashboard/INVALID"
    r = requests.get(url)
    assert r.status_code in (400, 404)

def test_invalid_prediction_payload():
    url = "http://localhost:8000/predict"
    r = requests.post(url, json={"bad": "data"})
    assert r.status_code in (400, 422)
