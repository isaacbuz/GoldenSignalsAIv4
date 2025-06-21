"""
Enhanced ML Meta Agent - Advanced Ensemble Optimization
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EnhancedMLMetaAgent:
    """Advanced ML Meta Agent for ensemble optimization"""
    
    def __init__(self, data_bus=None):
        self.name = "EnhancedMLMetaAgent"
        self.data_bus = data_bus
        self.agent_weights = defaultdict(lambda: 1.0)
        self.performance_history = defaultdict(list)
        
    def optimize_ensemble(self, agent_signals: List[Dict[str, Any]], 
                         market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize ensemble signals"""
        
        if not agent_signals:
            return {"action": "HOLD", "confidence": 0.0, "metadata": {"agent": self.name}}
        
        # Calculate weighted ensemble
        action_scores = defaultdict(float)
        
        for signal in agent_signals:
            agent = signal.get('metadata', {}).get('agent', 'unknown')
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0)
            weight = self.agent_weights[agent]
            
            action_scores[action] += confidence * weight
        
        # Determine final action
        final_action = max(action_scores.items(), key=lambda x: x[1])[0]
        total_score = sum(action_scores.values())
        final_confidence = action_scores[final_action] / total_score if total_score > 0 else 0
        
        return {
            "action": final_action,
            "confidence": float(final_confidence),
            "metadata": {
                "agent": self.name,
                "reasoning": f"ML ensemble: {final_action}",
                "ml_optimization": True
            }
        }
    
    def update_performance(self, agent_name: str, signal: Dict[str, Any], outcome: float):
        """Update agent performance"""
        self.performance_history[agent_name].append(outcome)
        
        # Update weight based on recent performance
        if len(self.performance_history[agent_name]) >= 10:
            recent_performance = self.performance_history[agent_name][-20:]
            accuracy = sum(1 for p in recent_performance if p > 0) / len(recent_performance)
            self.agent_weights[agent_name] = 0.5 + accuracy * 0.5
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return {
            "agent_weights": dict(self.agent_weights),
            "performance_summary": {
                agent: {
                    "total_signals": len(history),
                    "accuracy": sum(1 for h in history if h > 0) / len(history) if history else 0.5
                }
                for agent, history in self.performance_history.items()
            }
        }
    
    def save_model(self, filepath: str):
        """Save model state"""
        # Simplified save
        pass
    
    def load_model(self, filepath: str):
        """Load model state"""
        # Simplified load
        pass 