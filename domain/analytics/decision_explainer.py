class DecisionExplainer:
    def explain(self, decision):
        return {
            "symbol": decision.symbol,
            "action": decision.action.name,
            "confidence": decision.confidence,
            "rationale": decision.rationale
        }
