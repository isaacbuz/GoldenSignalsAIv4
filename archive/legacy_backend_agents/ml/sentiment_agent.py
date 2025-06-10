from archive.legacy_backend_agents.base import BaseAgent
import random

class SentimentAgent(BaseAgent):
    def run(self, market_data):
        # Dummy logic: random sentiment score
        score = random.randint(0, 100)
        if score > 60:
            signal = "positive"
            explanation = f"Market sentiment positive (score={score})"
        elif score < 40:
            signal = "negative"
            explanation = f"Market sentiment negative (score={score})"
        else:
            signal = "neutral"
            explanation = f"Market sentiment neutral (score={score})"
        return {
            "name": "Sentiment",
            "type": "ml",
            "signal": signal,
            "score": score,
            "confidence": 70,
            "explanation": explanation
        }

    def train(self, data):
        # Placeholder for training logic
        return "Sentiment model trained (dummy)"

