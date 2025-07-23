from typing import Dict, List

import numpy as np


class EnsembleSignalEngine:
    def __init__(self, models: List):
        self.models = models  # Each model must have .predict_proba

    def predict(self, features: List[float]) -> Dict:
        prob_accumulator = np.zeros(2)  # For binary classification: [hold, buy/sell]
        individual_signals = []

        for model in self.models:
            prob = model.predict_proba([features])[0]
            prob_accumulator += prob
            individual_signals.append(prob)

        avg_prob = prob_accumulator / len(self.models)
        signal = self._interpret(avg_prob)

        return {
            "signal": signal,
            "confidence": round(float(avg_prob[1]), 4),
            "raw_probs": [list(map(float, p)) for p in individual_signals],
            "method": "weighted_average"
        }

    def _interpret(self, avg_prob: np.ndarray) -> str:
        if avg_prob[1] > 0.6:
            return "buy"
        elif avg_prob[1] < 0.4:
            return "sell"
        return "hold"
