import numpy as np
import pickle
from archive.legacy_backend_agents.base import BaseSignalAgent

class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.rand(input_size, layer_size) * np.sqrt(1 / (input_size + layer_size)),
            np.random.rand(layer_size, output_size) * np.sqrt(1 / (layer_size + output_size)),
            np.zeros((1, layer_size)),
            np.zeros((1, output_size)),
        ]
    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-2]
        decision = np.dot(feed, self.weights[1]) + self.weights[-1]
        return decision
    def get_weights(self):
        return self.weights
    def set_weights(self, weights):
        self.weights = weights

class PretrainedEvolutionAgent(BaseSignalAgent):
    """
    Wraps the pre-trained evolutionary strategy model from Stock-Prediction-Models.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)
        # Load weights
        with open("external/Stock-Prediction-Models/realtime-agent/model.pkl", "rb") as f:
            self.weights = pickle.load(f)
        # Model architecture as per repo
        self.model = Model(input_size=40, layer_size=500, output_size=3)
        self.model.set_weights(self.weights)
    def run(self, price_history: list) -> dict:
        # Prepare input as per repo's get_state (window_size=20, 2 features: price and diff)
        if len(price_history) < 20:
            return {"agent": "PretrainedEvolutionAgent", "error": "Not enough data (need 20+ prices)"}
        # For demo: use price and price diff as two features
        price = np.array(price_history[-20:])
        diff = np.diff(price, prepend=price[0])
        inputs = np.concatenate([price, diff]).reshape(1, -1)  # shape (1, 40)
        logits = self.model.predict(inputs)
        action = int(np.argmax(logits))
        action_map = {0: "hold", 1: "buy", 2: "sell"}
        return {
            "agent": "PretrainedEvolutionAgent",
            "action": action_map[action],
            "logits": logits.tolist(),
            "confidence": float(np.max(logits)),
            "symbol": self.symbol
        }
