# base_strategy.py
# Freqtrade-style strategy architecture for GoldenSignalsAI

class BaseStrategy:
    def __init__(self, config=None):
        self.config = config or {}

    def indicators(self, data):
        """Compute raw indicators from price data"""
        return {
            "rsi": self.rsi(data["price"]),
            "macd": self.macd(data["price"])
        }

    def signal(self, indicators):
        """Generate trading signal from indicator values"""
        if indicators["rsi"] < 30 and indicators["macd"] > 0:
            return "buy"
        elif indicators["rsi"] > 70 and indicators["macd"] < 0:
            return "sell"
        else:
            return "hold"

    def rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
        deltas = [prices[i+1] - prices[i] for i in range(-period-1, -1)]
        gains = sum(d for d in deltas if d > 0)
        losses = sum(-d for d in deltas if d < 0)
        rs = gains / losses if losses > 0 else 1
        return 100 - (100 / (1 + rs))

    def macd(self, prices, fast=12, slow=26):
        ema_fast = self.ema(prices, fast)
        ema_slow = self.ema(prices, slow)
        return ema_fast - ema_slow

    def ema(self, prices, period):
        k = 2 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema
