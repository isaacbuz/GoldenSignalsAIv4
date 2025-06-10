from archive.legacy_backend_agents.base import BaseAgent

class AdaptiveOscillatorAgent(BaseAgent):
    def run(self, market_data):
        prices = market_data.get("price", [])
        if len(prices) < 15:
            return self.output("neutral", None, "Not enough data")
        # Example: combine RSI, MACD, CCI (simulate with price slices)
        rsi = prices[-14:] if len(prices) >= 14 else prices
        macd = prices[-14:] if len(prices) >= 14 else prices
        cci = prices[-14:] if len(prices) >= 14 else prices
        # Weighted average
        rsi_score = rsi[-1] if rsi else 0
        macd_score = macd[-1] if macd else 0
        cci_score = cci[-1] if cci else 0
        score = rsi_score * 0.3 + macd_score * 0.4 + cci_score * 0.3
        score = max(min(score, 100), -100)
        if score > 70:
            signal = "strong_buy"
            confidence = 90
        elif score > 50:
            signal = "buy"
            confidence = 80
        elif score < 30:
            signal = "sell"
            confidence = 80
        else:
            signal = "neutral"
            confidence = 60
        explanation = f"Adaptive Oscillator score: {score:.2f}"
        return self.output(signal, score, explanation)

    def output(self, signal, score, explanation):
        return {
            "name": "AdaptiveOscillator",
            "type": "indicator",
            "signal": signal,
            "score": score,
            "confidence": 80,
            "explanation": explanation
        }
