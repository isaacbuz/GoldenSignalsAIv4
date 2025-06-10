import numpy as np
from sklearn.mixture import GaussianMixture
from archive.legacy_backend_agents.base import BaseSignalAgent

class RegimeDetectionAgent(BaseSignalAgent):
    """
    Detects market regimes (bull, bear, sideways) using clustering on returns.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.model = GaussianMixture(n_components=3)

    def run(self, price_history: list) -> dict:
        if len(price_history) < 30:
            return {"agent": "RegimeDetectionAgent", "regime": None, "confidence": 0, "explanation": "Not enough data."}
        returns = np.diff(np.log(price_history))
        X = returns.reshape(-1, 1)
        self.model.fit(X)
        pred = self.model.predict([X[-1]])[0]
        regime_map = {0: "bull", 1: "bear", 2: "sideways"}
        regime = regime_map.get(pred, "unknown")
        explanation = f"Detected regime: {regime} (cluster {pred})"
        return {"agent": "RegimeDetectionAgent", "regime": regime, "confidence": 75, "explanation": explanation}
