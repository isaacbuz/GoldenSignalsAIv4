"""
Machine Learning Meta Agent
Learns from agent performance to dynamically adjust signal weights
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class MLMetaAgent:
    """
    ML-powered meta agent that learns optimal agent weights
    
    Features:
    - Dynamic weight adjustment based on performance
    - Agent correlation analysis
    - Market regime detection
    - Ensemble optimization
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.name = "MLMetaAgent"
        self.learning_rate = learning_rate
        self.agent_weights = defaultdict(lambda: 1.0)  # Start with equal weights
        self.performance_history = defaultdict(list)
        self.correlation_matrix = {}
        self.current_regime = "neutral"
        self.min_history = 20  # Minimum signals before adjusting weights
        
    def update_performance(self, agent_name: str, signal: Dict[str, Any], outcome: float):
        """Update agent performance history"""
        self.performance_history[agent_name].append({
            'signal': signal,
            'outcome': outcome,  # 1 for correct, -1 for wrong, 0 for neutral
            'timestamp': datetime.now(),
            'regime': self.current_regime
        })
        
        # Update weights if enough history
        if len(self.performance_history[agent_name]) >= self.min_history:
            self._update_agent_weight(agent_name)
    
    def _update_agent_weight(self, agent_name: str):
        """Update agent weight based on recent performance"""
        history = self.performance_history[agent_name][-50:]  # Last 50 signals
        
        # Calculate metrics
        total_signals = len(history)
        correct_signals = sum(1 for h in history if h['outcome'] > 0)
        accuracy = correct_signals / total_signals if total_signals > 0 else 0.5
        
        # Calculate regime-specific performance
        regime_performance = defaultdict(list)
        for h in history:
            regime_performance[h['regime']].append(h['outcome'])
        
        # Calculate Sharpe-like ratio (consistency)
        outcomes = [h['outcome'] for h in history]
        if len(outcomes) > 1:
            avg_outcome = np.mean(outcomes)
            std_outcome = np.std(outcomes)
            sharpe = avg_outcome / std_outcome if std_outcome > 0 else avg_outcome
        else:
            sharpe = 0
        
        # Update weight based on multiple factors
        # Base weight on accuracy
        new_weight = accuracy * 2  # Scale to 0-2
        
        # Adjust for consistency (Sharpe)
        if sharpe > 0.5:
            new_weight *= 1.2
        elif sharpe < -0.5:
            new_weight *= 0.8
        
        # Smooth weight update
        old_weight = self.agent_weights[agent_name]
        self.agent_weights[agent_name] = old_weight + self.learning_rate * (new_weight - old_weight)
        
        # Keep weights in reasonable range
        self.agent_weights[agent_name] = max(0.1, min(3.0, self.agent_weights[agent_name]))
    
    def detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Detect current market regime"""
        # Simplified regime detection based on volatility and trend
        volatility = market_data.get('volatility', 0.15)
        trend_strength = market_data.get('trend_strength', 0)
        
        if volatility > 0.25:
            if abs(trend_strength) > 0.5:
                return "volatile_trending"
            else:
                return "volatile_ranging"
        else:
            if abs(trend_strength) > 0.5:
                return "calm_trending"
            else:
                return "calm_ranging"
    
    def calculate_agent_correlations(self, signals: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate correlation between agent signals"""
        if len(signals) < 2:
            return {}
        
        # Extract actions as numeric values
        actions = []
        agent_names = []
        
        for signal in signals:
            agent = signal.get('agent', 'unknown')
            action = signal.get('action', 'NEUTRAL')
            
            # Convert to numeric
            if action == 'BUY':
                value = 1
            elif action == 'SELL':
                value = -1
            else:
                value = 0
            
            actions.append(value)
            agent_names.append(agent)
        
        # Calculate average correlation
        if len(set(actions)) == 1:  # All same action
            avg_correlation = 1.0
        else:
            # Simple correlation metric
            agreement = sum(1 for i in range(len(actions)-1) 
                          for j in range(i+1, len(actions)) 
                          if actions[i] == actions[j])
            total_pairs = len(actions) * (len(actions) - 1) / 2
            avg_correlation = agreement / total_pairs if total_pairs > 0 else 0
        
        return {'average_correlation': avg_correlation}
    
    def optimize_ensemble(self, agent_signals: List[Dict[str, Any]], 
                         market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize ensemble of agent signals using ML insights"""
        
        if not agent_signals:
            return self._create_signal("NEUTRAL", 0, "No agent signals")
        
        # Update market regime
        if market_data:
            self.current_regime = self.detect_market_regime(market_data)
        
        # Calculate correlations
        correlations = self.calculate_agent_correlations(agent_signals)
        
        # Weighted voting with ML-adjusted weights
        weighted_scores = defaultdict(float)
        total_weight = 0
        signal_details = []
        
        for signal in agent_signals:
            agent = signal.get('agent', 'unknown')
            action = signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0.5)
            
            # Get ML-adjusted weight
            weight = self.agent_weights[agent]
            
            # Adjust weight based on regime
            if self.current_regime == "volatile_ranging" and agent in ['trend_following', 'momentum']:
                weight *= 0.7  # Reduce trend-following in ranging markets
            elif self.current_regime == "calm_trending" and agent in ['mean_reversion', 'oscillator']:
                weight *= 0.7  # Reduce mean-reversion in trending markets
            
            # Apply weight
            if action == 'BUY':
                weighted_scores['BUY'] += confidence * weight
            elif action == 'SELL':
                weighted_scores['SELL'] += confidence * weight
            else:
                weighted_scores['NEUTRAL'] += confidence * weight
            
            total_weight += weight
            
            signal_details.append({
                'agent': agent,
                'action': action,
                'confidence': confidence,
                'weight': weight,
                'weighted_contribution': confidence * weight
            })
        
        # Normalize scores
        if total_weight > 0:
            for action in weighted_scores:
                weighted_scores[action] /= total_weight
        
        # Determine consensus action
        if not weighted_scores:
            consensus_action = 'NEUTRAL'
            consensus_confidence = 0
        else:
            # Get action with highest score
            consensus_action = max(weighted_scores.items(), key=lambda x: x[1])[0]
            consensus_confidence = weighted_scores[consensus_action]
            
            # Adjust confidence based on agreement
            avg_correlation = correlations.get('average_correlation', 0.5)
            if avg_correlation < 0.3:  # Low agreement
                consensus_confidence *= 0.8
                
        # Create enhanced consensus signal
        reasoning = self._generate_reasoning(signal_details, consensus_action, 
                                           correlations, self.current_regime)
        
        return self._create_signal(
            consensus_action,
            consensus_confidence,
            reasoning,
            {
                'weighted_scores': dict(weighted_scores),
                'agent_weights': dict(self.agent_weights),
                'correlations': correlations,
                'market_regime': self.current_regime,
                'signal_details': signal_details,
                'ml_optimization': True
            }
        )
    
    def _generate_reasoning(self, signal_details: List[Dict], action: str, 
                          correlations: Dict, regime: str) -> str:
        """Generate detailed reasoning for the ML-optimized signal"""
        # Count votes
        buy_agents = sum(1 for s in signal_details if s['action'] == 'BUY')
        sell_agents = sum(1 for s in signal_details if s['action'] == 'SELL')
        neutral_agents = sum(1 for s in signal_details if s['action'] == 'NEUTRAL')
        
        # Top contributors
        top_contributors = sorted(signal_details, 
                                key=lambda x: x['weighted_contribution'], 
                                reverse=True)[:3]
        
        # Build top signals string
        top_signals_str = ', '.join(f"{s['agent']} ({s['action']}:{s['confidence']:.2f})" 
                                   for s in top_contributors)
        
        reasoning_parts = [
            f"ML Meta-Agent consensus: {action}",
            f"Market regime: {regime}",
            f"Agent votes: {buy_agents} buy, {sell_agents} sell, {neutral_agents} neutral",
            f"Correlation: {correlations.get('average_correlation', 0):.2%}",
            f"Top signals: {top_signals_str}"
        ]
        
        return " | ".join(reasoning_parts)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        report = {
            'agent_weights': dict(self.agent_weights),
            'agent_performance': {},
            'regime_performance': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'total_signals_processed': sum(len(h) for h in self.performance_history.values())
        }
        
        # Calculate per-agent metrics
        for agent, history in self.performance_history.items():
            if not history:
                continue
                
            outcomes = [h['outcome'] for h in history]
            report['agent_performance'][agent] = {
                'total_signals': len(history),
                'accuracy': sum(1 for o in outcomes if o > 0) / len(outcomes),
                'avg_outcome': np.mean(outcomes),
                'recent_performance': np.mean([h['outcome'] for h in history[-10:]])
            }
            
            # Regime breakdown
            for h in history:
                regime = h['regime']
                report['regime_performance'][regime]['total'] += 1
                if h['outcome'] > 0:
                    report['regime_performance'][regime]['correct'] += 1
        
        # Calculate regime accuracies
        for regime, stats in report['regime_performance'].items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
        
        return report
    
    def save_model(self, filepath: str):
        """Save ML model state"""
        model_state = {
            'agent_weights': dict(self.agent_weights),
            'performance_history': {k: v[-100:] for k, v in self.performance_history.items()},  # Last 100
            'current_regime': self.current_regime,
            'learning_rate': self.learning_rate
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_state, f, indent=2, default=str)
    
    def load_model(self, filepath: str):
        """Load ML model state"""
        try:
            with open(filepath, 'r') as f:
                model_state = json.load(f)
            
            self.agent_weights = defaultdict(lambda: 1.0, model_state.get('agent_weights', {}))
            self.current_regime = model_state.get('current_regime', 'neutral')
            self.learning_rate = model_state.get('learning_rate', 0.01)
            
            # Load performance history
            for agent, history in model_state.get('performance_history', {}).items():
                self.performance_history[agent] = history
                
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
    
    def _create_signal(self, action: str, confidence: float, reason: str, 
                      metadata: Dict = None) -> Dict[str, Any]:
        """Create standardized signal output"""
        return {
            "agent": self.name,
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        } 