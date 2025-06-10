"""
Meta signal agent for combining signals from multiple agents.
"""

from typing import List, Dict
import numpy as np

class MetaSignalAgent:
    """
    An agent that combines signals from multiple agents using weighted voting.
    """
    def __init__(self, weight_config: Dict[str, float] = None):
        """
        Initialize the meta signal agent.
        
        Args:
            weight_config (Dict[str, float], optional): Weights for each agent type.
                Defaults to technical: 0.4, sentiment: 0.3, regime: 0.3
        """
        # Default weights for each agent type
        self.weights = weight_config or {
            "technical": 0.4,
            "sentiment": 0.3,
            "regime": 0.3
        }

    def predict(self, agent_signals: Dict[str, Dict]) -> Dict:
        """
        Combine signals from multiple agents using weighted voting.
        
        Args:
            agent_signals (Dict[str, Dict]): Signals from different agents.
                Format: { "technical": {"signal": "buy", "confidence": 0.8}, ... }
                
        Returns:
            Dict: Combined signal with format:
                {
                    "signal": str,  # "buy", "sell", or "hold"
                    "score": float,  # Confidence score
                    "details": Dict  # Vote scores for each signal
                }
        """
        vote_scores = {"buy": 0, "sell": 0, "hold": 0}

        for agent, data in agent_signals.items():
            signal = data["signal"]
            confidence = data.get("confidence", 0.5)
            weight = self.weights.get(agent, 0.0)
            vote_scores[signal] += confidence * weight

        final_signal = max(vote_scores, key=vote_scores.get)
        return {
            "signal": final_signal,
            "score": vote_scores[final_signal],
            "details": vote_scores
        } 