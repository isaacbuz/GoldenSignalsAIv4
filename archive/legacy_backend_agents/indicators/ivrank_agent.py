from archive.legacy_backend_agents.base import BaseAgent

class IVRankAgent(BaseAgent):
    def run(self, market_data):
        iv_history = market_data.get("iv_history", [])
        current_iv = market_data.get("current_iv", None)
        if not iv_history or current_iv is None:
            return self.output("neutral", None, "No IV data")
        min_iv = min(iv_history)
        max_iv = max(iv_history)
        iv_rank = 100 * (current_iv - min_iv) / (max_iv - min_iv) if max_iv > min_iv else 50
        if iv_rank > 70:
            signal = "sell_premium"
            confidence = 80
            explanation = f"IV Rank high ({iv_rank:.1f}): options are expensive."
        elif iv_rank < 30:
            signal = "buy_premium"
            confidence = 80
            explanation = f"IV Rank low ({iv_rank:.1f}): options are cheap."
        else:
            signal = "neutral"
            confidence = 60
            explanation = f"IV Rank moderate ({iv_rank:.1f}): options fairly priced."
        return self.output(signal, iv_rank, explanation)

    def output(self, signal, value, explanation):
        return {
            "name": "IVRank",
            "type": "indicator",
            "signal": signal,
            "iv_rank": value,
            "confidence": 80,
            "explanation": explanation
        }

