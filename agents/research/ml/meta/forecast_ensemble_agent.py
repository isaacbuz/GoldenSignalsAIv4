"""
Forecast Ensemble Agent that combines predictions from multiple forecasting models.
"""
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from .....base import BaseAgent

logger = logging.getLogger(__name__)

class ForecastEnsembleAgent(BaseAgent):
    """Agent that combines predictions from multiple forecasting models."""
    
    def __init__(self, models: List[BaseAgent] = None, voting_threshold: float = 0.6):
        """
        Initialize ensemble agent.
        
        Args:
            models: List of forecasting models to ensemble
            voting_threshold: Threshold for consensus (default 0.6 = 60% agreement)
        """
        super().__init__(name="ForecastEnsemble", agent_type="meta")
        self.models = models or []
        self.voting_threshold = voting_threshold
        
    def add_model(self, model: BaseAgent) -> None:
        """Add a new forecasting model to the ensemble."""
        if model not in self.models:
            self.models.append(model)
            
    def get_consensus(self, actions: List[str], confidences: List[float]) -> tuple[str, float]:
        """Calculate consensus action and confidence from model predictions."""
        if not actions:
            return "hold", 0.0
            
        # Count votes weighted by confidence
        votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        for action, confidence in zip(actions, confidences):
            votes[action] += confidence
            
        # Get total confidence and normalize votes
        total_confidence = sum(votes.values())
        if total_confidence > 0:
            votes = {k: v/total_confidence for k, v in votes.items()}
            
        # Find action with highest vote
        max_vote = max(votes.values())
        max_actions = [k for k, v in votes.items() if v == max_vote]
        
        # If clear winner with votes above threshold
        if len(max_actions) == 1 and max_vote >= self.voting_threshold:
            return max_actions[0], max_vote
        return "hold", 0.0
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data through all models and combine predictions."""
        if not self.models:
            logger.warning("No forecasting models configured")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": "No forecasting models configured"}
            }
            
        # Gather predictions from all models
        predictions = []
        for model in self.models:
            try:
                pred = model.process(data)
                if pred.get("action") != "hold":  # Only consider non-hold predictions
                    predictions.append(pred)
            except Exception as e:
                logger.error(f"Model {model.name} error: {e}")
                
        if not predictions:
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": "No valid predictions"}
            }
            
        # Extract actions and confidences
        actions = [p["action"] for p in predictions]
        confidences = [p["confidence"] for p in predictions]
        
        # Get consensus
        action, confidence = self.get_consensus(actions, confidences)
            
        return {
            "action": action,
            "confidence": confidence,
            "metadata": {
                "models": len(self.models),
                "valid_predictions": len(predictions),
                "model_predictions": [
                    {
                        "model": p.get("metadata", {}).get("model_name", "unknown"),
                        "action": p["action"],
                        "confidence": p["confidence"]
                    }
                    for p in predictions
                ],
                "voting_threshold": self.voting_threshold
            }
        } 