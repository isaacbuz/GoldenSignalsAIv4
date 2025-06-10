from archive.legacy_backend_agents.base import BaseSignalAgent

class EnsembleMetaAgent(BaseSignalAgent):
    """
    Aggregates outputs from multiple ML agents using a simple majority vote or weighted average.
    """
    def __init__(self, symbol: str, agents: list):
        super().__init__(symbol)
        self.agents = agents

    def run(self, market_data: dict) -> dict:
        results = [agent.run(market_data) for agent in self.agents]
        # Example: majority vote on 'signal' field
        signals = [r.get('signal') for r in results if 'signal' in r]
        if not signals:
            return {"agent": "EnsembleMetaAgent", "signal": None, "confidence": 0, "explanation": "No agent signals available."}
        signal = max(set(signals), key=signals.count)
        explanation = f"Ensemble consensus: {signal} from {len(results)} agents."
        return {"agent": "EnsembleMetaAgent", "signal": signal, "confidence": 90, "explanation": explanation, "details": results}
