from backend.agents.base import BaseAgent

class ClusteredSupportResistanceAgent(BaseAgent):
    def run(self, market_data):
        prices = market_data.get("price", [])
        if len(prices) < 10:
            return self.output("neutral", None, "Not enough data")
        sorted_prices = sorted(prices)
        clusters = [sorted_prices[5], sorted_prices[len(sorted_prices)//2], sorted_prices[-5]]
        explanation = f"Support/Resistance levels: {clusters}"
        return self.output("clustered_sr", clusters, explanation)

    def output(self, signal, value, explanation):
        return {
            "name": "ClusteredSupportResistance",
            "type": "indicator",
            "signal": signal,
            "value": value,
            "confidence": 65,
            "explanation": explanation
        }
