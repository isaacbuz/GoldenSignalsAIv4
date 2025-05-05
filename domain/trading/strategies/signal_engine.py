import pandas as pd

class SignalEngine:
    def __init__(self, data, weights=None):
        self.data = data
        self.weights = weights or {
            "ma_cross": 0.2,
            "ema_cross": 0.2,
            "vwap": 0.1,
            "bollinger": 0.2,
            "rsi": 0.15,
            "macd": 0.15
        }

    def compute_signal_score(self):
        return 0.75

    def generate_signal(self, symbol, risk_profile="balanced"):
        latest_price = self.data['close'].iloc[-1]
        return {
            "symbol": symbol,
            "action": "Buy",
            "price": latest_price,
            "confidence_score": 0.75,
            "stop_loss": latest_price * 0.98,
            "profit_target": latest_price * 1.04
        }
