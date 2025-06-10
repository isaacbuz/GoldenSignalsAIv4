class SmartMoneyAgent:
    def run(self, options_flow):
        if not options_flow or "trades" not in options_flow:
            return {"signal": "neutral", "confidence": 0.0, "explanation": "No data"}

        call_vol = sum(t["size"] for t in options_flow["trades"] if t["type"] == "call")
        put_vol = sum(t["size"] for t in options_flow["trades"] if t["type"] == "put")

        if call_vol > 1.2 * put_vol:
            return {
                "signal": "bullish",
                "confidence": min(call_vol / (put_vol + 1), 1.0),
                "explanation": "Call volume dominance"
            }
        elif put_vol > 1.2 * call_vol:
            return {
                "signal": "bearish",
                "confidence": min(put_vol / (call_vol + 1), 1.0),
                "explanation": "Put volume dominance"
            }
        else:
            return {
                "signal": "neutral",
                "confidence": 0.3,
                "explanation": "Mixed flow"
            }
