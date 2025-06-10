from archive.legacy_backend_agents.base import BaseAgent

class MLTrendChannelAgent(BaseAgent):
    def run(self, market_data):
        prices = market_data.get("price", [])
        if len(prices) < 10:
            return self.output("neutral", None, "Not enough data")
        mean = sum(prices) / len(prices)
        deviations = [abs(p - mean) for p in prices]
        stdev = sum(deviations) / len(prices)
        support = mean - stdev
        resistance = mean + stdev
        mid = mean
        explanation = f"Trend channel: support={support:.2f}, resistance={resistance:.2f}, mid={mid:.2f}"
        return self.output("trend_channel", {"support": support, "resistance": resistance, "mid": mid}, explanation)

    def output(self, signal, value, explanation):
        return {
            "name": "MLTrendChannel",
            "type": "indicator",
            "signal": signal,
            "value": value,
            "confidence": 70,
            "explanation": explanation
        }
