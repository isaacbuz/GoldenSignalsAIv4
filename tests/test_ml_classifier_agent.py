import numpy as np
import pandas as pd
from agents.ml_classifier_agent import MLClassifierAgent

def test_ml_classifier_predict():
    agent = MLClassifierAgent()
    features = pd.DataFrame(np.random.rand(5, 3))
    agent.train(features, pd.Series([0, 1, 0, 1, 0]))
    signal = agent.predict_signal(features)
    assert signal in ["buy", "sell", "hold"]
