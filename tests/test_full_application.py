import pytest
import requests
import numpy as np
import pandas as pd

def test_full_app_workflow():
    # Simulate a full application workflow: data in, API call, result out
    # 1. Prepare dummy data
    symbol = 'AAPL'
    features = list(np.random.rand(4))
    # 2. Call prediction endpoint
    url = 'http://localhost:8000/predict'
    response = requests.post(url, json={'symbol': symbol, 'features': features})
    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    # 3. Call dashboard endpoint
    url2 = f'http://localhost:8000/dashboard/{symbol}'
    response2 = requests.get(url2)
    assert response2.status_code == 200
    dash = response2.json()
    assert 'symbol' in dash and dash['symbol'] == symbol
    assert 'options_data' in dash
    # 4. Optionally, call health endpoint
    url3 = 'http://localhost:8000/health'
    response3 = requests.get(url3)
    assert response3.status_code == 200
    assert response3.json().get('status') == 'ok'
