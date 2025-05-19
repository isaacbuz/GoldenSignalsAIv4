import pytest
import numpy as np

# Dummy model for illustration; replace with actual imports
class DummyModel:
    def predict(self, X):
        return np.zeros(len(X))

def test_dummy_model_predict():
    model = DummyModel()
    X = np.random.rand(5, 3)
    preds = model.predict(X)
    assert (preds == 0).all()
    assert preds.shape == (5,)
