from backend.agents.base import BaseAgent

class KNNTrendAgent(BaseAgent):
    def run(self, market_data):
        prices = market_data.get("price", [])
        k = 5
        n = len(prices)
        if n < k + 1:
            return self.output("neutral", None, "Not enough data")
        recent = prices[-1]
        diffs = [abs(p - recent) for p in prices[-k-1:-1]]
        avg = sum(diffs) / k
        if recent > prices[-2] + avg:
            signal = "up"
            confidence = 75
            explanation = f"KNN: Uptrend detected (recent > avg diff)"
        elif recent < prices[-2] - avg:
            signal = "down"
            confidence = 75
            explanation = f"KNN: Downtrend detected (recent < avg diff)"
        else:
            signal = "neutral"
            confidence = 60
            explanation = f"KNN: No clear trend"
        return self.output(signal, recent, explanation)

    def output(self, signal, value, explanation):
        return {
            "name": "KNNTrendClassifier",
            "type": "indicator",
            "signal": signal,
            "value": value,
            "confidence": 75,
            "explanation": explanation
        }
