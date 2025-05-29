from backend.agents.base import BaseAgent

class RSIAgent(BaseAgent):
    def run(self, market_data):
        prices = market_data.get("price", [])
        if len(prices) < 15:
            return self.output("neutral", None, "Not enough data")

        gains, losses = [], []
        for i in range(1, 15):
            delta = prices[-i] - prices[-i-1]
            if delta > 0:
                gains.append(delta)
            else:
                losses.append(abs(delta))

        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        rs = avg_gain / avg_loss if avg_loss else 0
        rsi = 100 - (100 / (1 + rs))

        signal = "buy" if rsi < 30 else "sell" if rsi > 70 else "neutral"
        return self.output(signal, rsi, f"RSI={rsi:.2f}")

    def output(self, signal, value, explanation):
        return {
            "name": "RSI",
            "type": "indicator",
            "signal": signal,
            "value": value,
            "confidence": 70,
            "explanation": explanation
        }

