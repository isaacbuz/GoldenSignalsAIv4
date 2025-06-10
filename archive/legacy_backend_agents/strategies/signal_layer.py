# signal_layer.py
# OctoBot-style signal generation pipeline: raw → filtered → confirmed

class SignalLayer:
    def __init__(self, rsi=None, macd=None, threshold=70):
        self.rsi = rsi
        self.macd = macd
        self.threshold = threshold

    def raw_signal(self):
        return {
            "rsi": self.rsi,
            "macd": self.macd,
            "trend": "bullish" if self.macd > 0 else "bearish"
        }

    def filtered_signal(self):
        if self.rsi < 30 and self.macd > 0:
            return "potential_buy"
        elif self.rsi > 70 and self.macd < 0:
            return "potential_sell"
        return "neutral"

    def confirmed_signal(self):
        # High-confidence confirmation if RSI < 25 and MACD > 0.5
        if self.rsi < 25 and self.macd > 0.5:
            return "confirmed_buy"
        elif self.rsi > 75 and self.macd < -0.5:
            return "confirmed_sell"
        return "neutral"
