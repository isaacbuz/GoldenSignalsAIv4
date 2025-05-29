from typing import List, Type
from backend.agents.base import BaseSignalAgent

"""
AgentManager dynamically discovers and manages all agent classes that inherit from BaseSignalAgent.
No need to manually update ALL_AGENTS; simply create a new agent subclass and it will be registered automatically.
"""

class AgentManager:
    """
    Dynamically discovers and manages all agent classes that inherit from BaseSignalAgent.
    Supports advanced ML, NLP, and multimodal agents by passing appropriate data to each agent's run method.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.agents: List[BaseSignalAgent] = [cls(symbol) for cls in BaseSignalAgent.registry]

    def run_all(self, market_data=None, price_history=None, news_headlines=None, images=None, context=None, question=None, texts=None, returns_matrix=None, performance_metrics=None) -> List[dict]:
        """
        Run all agents and collect their signals. Passes relevant data to each agent based on its run() signature.
        """
        results = []
        for agent in self.agents:
            run_args = {}
            # Inspect agent signature and provide matching data
            sig = agent.run.__code__.co_varnames
            if 'market_data' in sig and market_data is not None:
                run_args['market_data'] = market_data
            if 'price_history' in sig and price_history is not None:
                run_args['price_history'] = price_history
            if 'news_headlines' in sig and news_headlines is not None:
                run_args['news_headlines'] = news_headlines
            if 'images' in sig and images is not None:
                run_args['images'] = images
            if 'context' in sig and context is not None:
                run_args['context'] = context
            if 'question' in sig and question is not None:
                run_args['question'] = question
            if 'texts' in sig and texts is not None:
                run_args['texts'] = texts
            if 'returns_matrix' in sig and returns_matrix is not None:
                run_args['returns_matrix'] = returns_matrix
            if 'performance_metrics' in sig and performance_metrics is not None:
                run_args['performance_metrics'] = performance_metrics
            try:
                result = agent.run(**run_args) if run_args else agent.run()
                results.append(result)
            except Exception as e:
                results.append({"agent": agent.__class__.__name__, "error": str(e)})
        return results

    def vote(self, market_data=None, price_history=None, news_headlines=None, images=None, context=None, question=None, texts=None, returns_matrix=None, performance_metrics=None) -> dict:
        """
        Aggregate agent signals and compute consensus action.
        """
        signals = self.run_all(market_data, price_history, news_headlines, images, context, question, texts, returns_matrix, performance_metrics)
        # Only consider agents that output an 'action' field
        votes = [s['action'] for s in signals if 'action' in s]
        consensus = max(set(votes), key=votes.count) if votes else None
        confidence = (
            sum(s['confidence'] for s in signals if s.get('action') == consensus and 'confidence' in s) / len(signals)
            if signals and consensus else 0
        )
        return {
            "symbol": self.symbol,
            "action": consensus,
            "confidence": round(confidence, 2),
            "sources": [s.get('agent', agent.__class__.__name__) for s, agent in zip(signals, self.agents) if s.get('action') == consensus]
        }

