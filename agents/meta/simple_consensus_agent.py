"""
Simple Meta Consensus Agent
Combines signals from multiple agents to generate a consensus signal
"""

from typing import Dict, Any, List
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SimpleConsensusAgent:
    """Combines signals from multiple agents using weighted voting"""
    
    def __init__(self):
        self.name = "simple_consensus_agent"
        
    def combine_signals(self, agent_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple agent signals into a consensus
        
        Args:
            agent_signals: List of signal dicts from different agents
            
        Returns:
            Consensus signal dictionary
        """
        if not agent_signals:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {
                    "agent": self.name,
                    "error": "No agent signals provided"
                }
            }
        
        # Extract valid signals (not errors)
        valid_signals = [s for s in agent_signals if s.get("confidence", 0) > 0]
        
        if not valid_signals:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {
                    "agent": self.name,
                    "error": "No valid signals available",
                    "total_agents": len(agent_signals)
                }
            }
        
        # Count votes weighted by confidence
        buy_weight = 0
        sell_weight = 0
        hold_weight = 0
        
        agent_details = []
        
        for signal in valid_signals:
            action = signal.get("action", "HOLD")
            confidence = signal.get("confidence", 0)
            agent_name = signal.get("metadata", {}).get("agent", "unknown")
            
            if action == "BUY":
                buy_weight += confidence
            elif action == "SELL":
                sell_weight += confidence
            else:
                hold_weight += confidence
                
            agent_details.append({
                "agent": agent_name,
                "action": action,
                "confidence": confidence
            })
        
        # Determine consensus action
        total_weight = buy_weight + sell_weight + hold_weight
        
        if total_weight == 0:
            consensus_action = "HOLD"
            consensus_confidence = 0.0
            reasoning = "No confident signals from agents"
        else:
            # Normalize weights
            buy_pct = buy_weight / total_weight
            sell_pct = sell_weight / total_weight
            hold_pct = hold_weight / total_weight
            
            # Determine action based on majority
            if buy_pct > 0.5:
                consensus_action = "BUY"
                consensus_confidence = buy_pct
                reasoning = f"Consensus BUY: {buy_pct:.0%} weighted agreement"
            elif sell_pct > 0.5:
                consensus_action = "SELL"
                consensus_confidence = sell_pct
                reasoning = f"Consensus SELL: {sell_pct:.0%} weighted agreement"
            elif buy_pct > sell_pct and buy_pct > hold_pct:
                consensus_action = "BUY"
                consensus_confidence = buy_pct * 0.8  # Lower confidence for non-majority
                reasoning = f"Weak BUY: {buy_pct:.0%} weighted preference"
            elif sell_pct > buy_pct and sell_pct > hold_pct:
                consensus_action = "SELL"
                consensus_confidence = sell_pct * 0.8
                reasoning = f"Weak SELL: {sell_pct:.0%} weighted preference"
            else:
                consensus_action = "HOLD"
                consensus_confidence = hold_pct
                reasoning = f"No clear consensus: BUY={buy_pct:.0%}, SELL={sell_pct:.0%}"
        
        # Calculate agreement score
        actions = [s.get("action") for s in valid_signals]
        most_common_action = max(set(actions), key=actions.count)
        agreement_score = actions.count(most_common_action) / len(actions)
        
        return {
            "action": consensus_action,
            "confidence": float(consensus_confidence),
            "metadata": {
                "agent": self.name,
                "reasoning": reasoning,
                "timestamp": datetime.now().isoformat(),
                "consensus_details": {
                    "buy_weight": float(buy_weight),
                    "sell_weight": float(sell_weight),
                    "hold_weight": float(hold_weight),
                    "agreement_score": float(agreement_score),
                    "total_agents": len(agent_signals),
                    "valid_agents": len(valid_signals)
                },
                "agent_votes": agent_details
            }
        } 