import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

class AutoAdaptiveAgent:
    def __init__(self, model_path="adaptive_agent.pkl", retrain_interval=100):
        self.model_path = model_path
        self.retrain_interval = retrain_interval
        self.signal_history = []
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = RandomForestClassifier(n_estimators=50)

    def save_model(self):
        joblib.dump(self.model, self.model_path)

    def tune_thresholds(self):
        # Placeholder: could adjust based on recent volatility or model confidence scores
        return {"buy": 0.6, "sell": 0.4}

    def retrain(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        self.save_model()

    def predict(self, features: np.ndarray):
        probs = self.model.predict_proba([features])[0]
        thresholds = self.tune_thresholds()
        if probs[1] > thresholds["buy"]:
            return "buy"
        elif probs[1] < thresholds["sell"]:
            return "sell"
        else:
            return "hold"

    def observe_and_learn(self, features: np.ndarray, outcome: int):
        self.signal_history.append((features, outcome))
        if len(self.signal_history) >= self.retrain_interval:
            X, y = zip(*self.signal_history)
            self.retrain(pd.DataFrame(X), pd.Series(y))
            self.signal_history.clear()
