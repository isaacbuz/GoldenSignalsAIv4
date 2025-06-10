class ApprovalAgent:
    def __init__(self, min_confidence=0.65, allowed_regimes=None):
        self.min_confidence = min_confidence
        self.allowed_regimes = allowed_regimes or ["normal", "low_volatility"]

    def run(self, signal, context):
        regime = context.get("regime")
        override = context.get("override", {})
        smart_money = context.get("smart_money", {})

        if override.get("override"):
            return {
                "accepted": True,
                "final_signal": override["override"],
                "reason": "Overridden by sentiment/news"
            }

        if signal["confidence"] < self.min_confidence:
            return {
                "accepted": False,
                "reason": f"Confidence {signal['confidence']} < {self.min_confidence}"
            }

        if regime not in self.allowed_regimes:
            return {
                "accepted": False,
                "reason": f"Blocked in regime: {regime}"
            }

        if smart_money and smart_money.get("signal") != signal["signal"] and smart_money.get("confidence", 0) > 0.7:
            return {
                "accepted": False,
                "reason": "Smart money conflict"
            }

        return {
            "accepted": True,
            "final_signal": signal["signal"],
            "reason": "Passes approval"
        }
