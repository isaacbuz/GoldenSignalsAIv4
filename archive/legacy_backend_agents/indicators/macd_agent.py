from archive.legacy_backend_agents.base import BaseAgent

class MACDAgent(BaseAgent):
    def run(self, market_data):
        prices = market_data.get("price", [])
        if len(prices) < 35:
            return self.output("neutral", None, "Not enough data")

        macd_line = self.ema(prices, 12) - self.ema(prices, 26)
        signal_line = self.ema([self.ema(prices, 12) - self.ema(prices, 26) for i in range(len(prices))], 9)

        if macd_line > signal_line:
            signal = "buy"
            confidence = 75
            explanation = f"MACD line ({macd_line:.2f}) > Signal line ({signal_line:.2f}): bullish momentum."
        elif macd_line < signal_line:
            signal = "sell"
            confidence = 75
            explanation = f"MACD line ({macd_line:.2f}) < Signal line ({signal_line:.2f}): bearish momentum."
        else:
            signal = "neutral"
            confidence = 50
            explanation = "MACD line equals Signal line: neutral momentum."
        return self.output(signal, macd_line, explanation, signal_line)

    def output(self, signal, macd, explanation, signal_line=None):
        return {
            "name": "MACD",
            "type": "indicator",
            "signal": signal,
            "macd": macd,
            "signal_line": signal_line,
            "confidence": 75,
            "explanation": explanation
        }

    def ema(self, prices, period):
        if len(prices) < period:
            return sum(prices) / len(prices)
        k = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * k + ema * (1 - k)
        return ema

