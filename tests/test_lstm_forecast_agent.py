import numpy as np
import pandas as pd
from agents.lstm_forecast_agent import LSTMForecastAgent

def test_lstm_forecast_predict():
    agent = LSTMForecastAgent()
    series = pd.Series(np.linspace(100, 110, 30))
    prediction = agent.predict(series)
    assert isinstance(prediction, float)
