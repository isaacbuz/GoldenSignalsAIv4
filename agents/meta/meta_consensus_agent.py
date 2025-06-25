"""
Meta Consensus Agent - Combines signals from multiple agents using ensemble methods.
Implements weighted voting, Bayesian updating, and confidence-based consensus.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from scipy import stats
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from src.common.base.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class MetaConsensusAgent(BaseAgent):
    """Meta agent that combines signals from multiple agents into consensus signals."""
    
    def __init__(
        self,
        name: str = "MetaConsensus",
        min_agents_for_consensus: int = 3,
        confidence_weight_factor: float = 2.0,
        recent_performance_weight: float = 0.3,
        agent_type_weights: Optional[Dict[str, float]] = None,
        consensus_threshold: float = 0.6,
        uncertainty_threshold: float = 0.3
    ):
        """
        Initialize Meta Consensus agent.
        
        Args:
            name: Agent name
            min_agents_for_consensus: Minimum agents needed for consensus
            confidence_weight_factor: How much to weight agent confidence
            recent_performance_weight: Weight for recent agent performance
            agent_type_weights: Weights for different agent types
            consensus_threshold: Threshold for strong consensus
            uncertainty_threshold: Threshold for high uncertainty
        """
        super().__init__(name=name, agent_type="meta")
        self.min_agents_for_consensus = min_agents_for_consensus
        self.confidence_weight_factor = confidence_weight_factor
        self.recent_performance_weight = recent_performance_weight
        self.consensus_threshold = consensus_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # Default weights for different agent types
        self.agent_type_weights = agent_type_weights or {
            'technical': 1.0,
            'fundamental': 1.2,
            'sentiment': 0.8,
            'macro': 1.1,
            'volatility': 0.9,
            'volume': 0.9,
            'options': 0.8,
            'flow': 1.0,
            'arbitrage': 1.3,
            'ml': 1.1
        }
        
        # Track agent performance for adaptive weighting
        self.agent_performance_history = {}
        
    def calculate_agent_weights(self, agent_signals: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate weights for each agent based on type, confidence, and performance."""
        try:
            weights = {}
            
            for signal in agent_signals:
                agent_name = signal.get('agent_name', 'unknown')
                agent_type = signal.get('agent_type', 'unknown')
                confidence = signal.get('confidence', 0.5)
                
                # Base weight from agent type
                base_weight = self.agent_type_weights.get(agent_type, 1.0)
                
                # Confidence weighting
                confidence_weight = 1.0 + (confidence - 0.5) * self.confidence_weight_factor
                
                # Performance-based weight
                performance_weight = 1.0
                if agent_name in self.agent_performance_history:
                    recent_performance = self.agent_performance_history[agent_name].get('recent_accuracy', 0.5)
                    performance_weight = 0.5 + recent_performance
                
                # Combined weight
                total_weight = base_weight * confidence_weight * performance_weight
                weights[agent_name] = max(0.1, min(3.0, total_weight))  # Clamp weights
            
            return weights
            
        except Exception as e:
            logger.error(f"Agent weight calculation failed: {str(e)}")
            return {signal.get('agent_name', f'agent_{i}'): 1.0 for i, signal in enumerate(agent_signals)}
    
    def weighted_voting_consensus(self, agent_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus using weighted voting."""
        try:
            if len(agent_signals) < self.min_agents_for_consensus:
                return {
                    'consensus_action': 'hold',
                    'consensus_confidence': 0.0,
                    'consensus_method': 'insufficient_agents',
                    'agent_count': len(agent_signals)
                }
            
            # Calculate weights
            weights = self.calculate_agent_weights(agent_signals)
            
            # Aggregate votes
            buy_weight = 0.0
            sell_weight = 0.0
            hold_weight = 0.0
            total_weight = 0.0
            
            agent_votes = []
            
            for signal in agent_signals:
                agent_name = signal.get('agent_name', 'unknown')
                action = signal.get('action', 'hold').lower()
                confidence = signal.get('confidence', 0.5)
                weight = weights.get(agent_name, 1.0)
                
                # Weight the vote by confidence and agent weight
                vote_strength = weight * confidence
                
                if action == 'buy':
                    buy_weight += vote_strength
                elif action == 'sell':
                    sell_weight += vote_strength
                else:
                    hold_weight += vote_strength
                
                total_weight += weight
                
                agent_votes.append({
                    'agent': agent_name,
                    'action': action,
                    'confidence': confidence,
                    'weight': weight,
                    'vote_strength': vote_strength
                })
            
            # Normalize weights
            if total_weight > 0:
                buy_ratio = buy_weight / total_weight
                sell_ratio = sell_weight / total_weight
                hold_ratio = hold_weight / total_weight
            else:
                buy_ratio = sell_ratio = hold_ratio = 0.33
            
            # Determine consensus action
            max_ratio = max(buy_ratio, sell_ratio, hold_ratio)
            
            if max_ratio == buy_ratio and buy_ratio > self.consensus_threshold:
                consensus_action = 'buy'
                consensus_confidence = buy_ratio
            elif max_ratio == sell_ratio and sell_ratio > self.consensus_threshold:
                consensus_action = 'sell'
                consensus_confidence = sell_ratio
            elif max_ratio == hold_ratio and hold_ratio > self.consensus_threshold:
                consensus_action = 'hold'
                consensus_confidence = hold_ratio
            else:
                # No strong consensus
                if max_ratio == buy_ratio:
                    consensus_action = 'buy'
                elif max_ratio == sell_ratio:
                    consensus_action = 'sell'
                else:
                    consensus_action = 'hold'
                consensus_confidence = max_ratio * 0.7  # Reduce confidence for weak consensus
            
            # Calculate consensus strength
            if max_ratio > 0.8:
                consensus_strength = 'very_strong'
            elif max_ratio > 0.6:
                consensus_strength = 'strong'
            elif max_ratio > 0.4:
                consensus_strength = 'moderate'
            else:
                consensus_strength = 'weak'
            
            return {
                'consensus_action': consensus_action,
                'consensus_confidence': consensus_confidence,
                'consensus_strength': consensus_strength,
                'consensus_method': 'weighted_voting',
                'vote_distribution': {
                    'buy_ratio': buy_ratio,
                    'sell_ratio': sell_ratio,
                    'hold_ratio': hold_ratio
                },
                'agent_votes': agent_votes,
                'total_weight': total_weight
            }
            
        except Exception as e:
            logger.error(f"Weighted voting consensus failed: {str(e)}")
            return {
                'consensus_action': 'hold',
                'consensus_confidence': 0.0,
                'consensus_method': 'error',
                'error': str(e)
            }
    
    def bayesian_consensus(self, agent_signals: List[Dict[str, Any]], prior_beliefs: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate consensus using Bayesian updating."""
        try:
            if len(agent_signals) < 2:
                return {'bayesian_consensus': 'insufficient_data'}
            
            # Prior beliefs (can be based on market conditions, recent performance, etc.)
            if prior_beliefs is None:
                prior_beliefs = {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
            
            # Start with prior
            posterior = prior_beliefs.copy()
            
            # Calculate agent weights
            weights = self.calculate_agent_weights(agent_signals)
            
            # Update posterior with each agent's signal
            for signal in agent_signals:
                agent_name = signal.get('agent_name', 'unknown')
                action = signal.get('action', 'hold').lower()
                confidence = signal.get('confidence', 0.5)
                weight = weights.get(agent_name, 1.0)
                
                # Calculate likelihood based on agent confidence and historical accuracy
                agent_accuracy = self.agent_performance_history.get(agent_name, {}).get('historical_accuracy', 0.6)
                
                # Likelihood of agent giving this signal if it's correct
                if action == 'buy':
                    likelihood = {'buy': confidence * agent_accuracy, 'sell': (1-confidence) * (1-agent_accuracy), 'hold': 0.1}
                elif action == 'sell':
                    likelihood = {'sell': confidence * agent_accuracy, 'buy': (1-confidence) * (1-agent_accuracy), 'hold': 0.1}
                else:
                    likelihood = {'hold': confidence * agent_accuracy, 'buy': (1-confidence) * (1-agent_accuracy) / 2, 'sell': (1-confidence) * (1-agent_accuracy) / 2}
                
                # Bayesian update with weight
                normalization = sum(posterior[outcome] * likelihood[outcome] for outcome in posterior.keys())
                
                if normalization > 0:
                    for outcome in posterior.keys():
                        # Weighted update
                        update_factor = 1.0 + (weight - 1.0) * self.recent_performance_weight
                        posterior[outcome] = (posterior[outcome] * likelihood[outcome] * update_factor) / normalization
            
            # Normalize posterior
            total_posterior = sum(posterior.values())
            if total_posterior > 0:
                for outcome in posterior.keys():
                    posterior[outcome] /= total_posterior
            
            # Determine consensus
            max_posterior = max(posterior.values())
            consensus_action = max(posterior, key=posterior.get)
            
            # Calculate certainty (entropy-based)
            entropy = -sum(p * np.log(p) for p in posterior.values() if p > 0)
            max_entropy = np.log(len(posterior))
            certainty = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
            
            return {
                'bayesian_action': consensus_action,
                'bayesian_confidence': max_posterior * certainty,
                'posterior_distribution': posterior,
                'certainty': certainty,
                'entropy': entropy
            }
            
        except Exception as e:
            logger.error(f"Bayesian consensus failed: {str(e)}")
            return {'bayesian_consensus': 'error'}
    
    def confidence_weighted_average(self, agent_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus using confidence-weighted averaging."""
        try:
            if not agent_signals:
                return {'confidence_consensus': 'no_signals'}
            
            # Convert actions to numerical values for averaging
            action_values = {'buy': 1.0, 'hold': 0.0, 'sell': -1.0}
            
            weighted_sum = 0.0
            weight_sum = 0.0
            confidence_sum = 0.0
            
            for signal in agent_signals:
                action = signal.get('action', 'hold').lower()
                confidence = signal.get('confidence', 0.5)
                
                action_value = action_values.get(action, 0.0)
                
                # Weight by confidence squared to emphasize high-confidence signals
                weight = confidence ** self.confidence_weight_factor
                
                weighted_sum += action_value * weight
                weight_sum += weight
                confidence_sum += confidence
            
            if weight_sum > 0:
                average_action_value = weighted_sum / weight_sum
                average_confidence = confidence_sum / len(agent_signals)
                
                # Convert back to action
                if average_action_value > 0.3:
                    consensus_action = 'buy'
                elif average_action_value < -0.3:
                    consensus_action = 'sell'
                else:
                    consensus_action = 'hold'
                
                # Adjust confidence based on agreement
                action_variance = np.var([action_values.get(s.get('action', 'hold').lower(), 0.0) for s in agent_signals])
                agreement_factor = 1.0 - (action_variance / 1.0)  # Max variance is 1.0
                
                final_confidence = average_confidence * agreement_factor
                
                return {
                    'confidence_action': consensus_action,
                    'confidence_score': final_confidence,
                    'average_action_value': average_action_value,
                    'agreement_factor': agreement_factor,
                    'action_variance': action_variance
                }
            
            return {'confidence_consensus': 'no_valid_weights'}
            
        except Exception as e:
            logger.error(f"Confidence weighted average failed: {str(e)}")
            return {'confidence_consensus': 'error'}
    
    def detect_signal_conflicts(self, agent_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect and analyze conflicts between agent signals."""
        try:
            if len(agent_signals) < 2:
                return {'conflicts': 'insufficient_signals'}
            
            actions = [signal.get('action', 'hold').lower() for signal in agent_signals]
            action_counts = {action: actions.count(action) for action in set(actions)}
            
            # Analyze disagreement
            total_signals = len(agent_signals)
            unique_actions = len(set(actions))
            
            # Calculate conflict metrics
            if unique_actions == 1:
                conflict_level = 'no_conflict'
                disagreement_ratio = 0.0
            elif unique_actions == 2:
                conflict_level = 'moderate_conflict'
                min_count = min(action_counts.values())
                disagreement_ratio = min_count / total_signals
            else:
                conflict_level = 'high_conflict'
                max_count = max(action_counts.values())
                disagreement_ratio = (total_signals - max_count) / total_signals
            
            # Identify conflicting agents
            conflicting_pairs = []
            for i, signal1 in enumerate(agent_signals):
                for j, signal2 in enumerate(agent_signals[i+1:], i+1):
                    action1 = signal1.get('action', 'hold').lower()
                    action2 = signal2.get('action', 'hold').lower()
                    
                    if (action1 == 'buy' and action2 == 'sell') or (action1 == 'sell' and action2 == 'buy'):
                        conflicting_pairs.append({
                            'agent1': signal1.get('agent_name', f'agent_{i}'),
                            'agent2': signal2.get('agent_name', f'agent_{j}'),
                            'action1': action1,
                            'action2': action2,
                            'confidence1': signal1.get('confidence', 0.5),
                            'confidence2': signal2.get('confidence', 0.5)
                        })
            
            return {
                'conflict_level': conflict_level,
                'disagreement_ratio': disagreement_ratio,
                'action_distribution': action_counts,
                'conflicting_pairs': conflicting_pairs,
                'unique_actions': unique_actions
            }
            
        except Exception as e:
            logger.error(f"Signal conflict detection failed: {str(e)}")
            return {'conflicts': 'error'}
    
    def generate_meta_consensus(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final consensus signal using multiple methods."""
        try:
            agent_signals = data.get('agent_signals', [])
            market_context = data.get('market_context', {})
            
            if len(agent_signals) < self.min_agents_for_consensus:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'signal_type': 'insufficient_agents',
                    'reasoning': f'Only {len(agent_signals)} agents, need {self.min_agents_for_consensus}'
                }
            
            # Apply multiple consensus methods
            weighted_consensus = self.weighted_voting_consensus(agent_signals)
            bayesian_consensus = self.bayesian_consensus(agent_signals)
            confidence_consensus = self.confidence_weighted_average(agent_signals)
            conflict_analysis = self.detect_signal_conflicts(agent_signals)
            
            # Combine results from different methods
            primary_action = weighted_consensus.get('consensus_action', 'hold')
            primary_confidence = weighted_consensus.get('consensus_confidence', 0.0)
            
            # Bayesian confirmation
            bayesian_action = bayesian_consensus.get('bayesian_action', 'hold')
            bayesian_confidence = bayesian_consensus.get('bayesian_confidence', 0.0)
            
            # Confidence-weighted confirmation
            confidence_action = confidence_consensus.get('confidence_action', 'hold')
            confidence_score = confidence_consensus.get('confidence_score', 0.0)
            
            # Check for method agreement
            methods_agree = (primary_action == bayesian_action == confidence_action)
            
            if methods_agree:
                final_action = primary_action
                final_confidence = (primary_confidence + bayesian_confidence + confidence_score) / 3
                consensus_type = 'strong_consensus'
            else:
                # Weighted combination when methods disagree
                method_weights = {'weighted': 0.4, 'bayesian': 0.3, 'confidence': 0.3}
                
                action_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
                action_scores[primary_action] += method_weights['weighted'] * primary_confidence
                action_scores[bayesian_action] += method_weights['bayesian'] * bayesian_confidence
                action_scores[confidence_action] += method_weights['confidence'] * confidence_score
                
                final_action = max(action_scores, key=action_scores.get)
                final_confidence = action_scores[final_action] * 0.8  # Reduce confidence for disagreement
                consensus_type = 'mixed_consensus'
            
            # Conflict adjustments
            conflict_level = conflict_analysis.get('conflict_level', 'no_conflict')
            if conflict_level == 'high_conflict':
                final_confidence *= 0.6
                consensus_type = 'uncertain_consensus'
            elif conflict_level == 'moderate_conflict':
                final_confidence *= 0.8
            
            # Market context adjustments
            volatility = market_context.get('volatility', 'normal')
            if volatility == 'high':
                final_confidence *= 0.9
            
            reasoning = [
                f"{consensus_type} from {len(agent_signals)} agents",
                f"Primary method: {weighted_consensus.get('consensus_strength', 'unknown')} weighted consensus",
                f"Conflict level: {conflict_level}"
            ]
            
            if methods_agree:
                reasoning.append("All consensus methods agree")
            else:
                reasoning.append("Mixed signals across consensus methods")
            
            return {
                'action': final_action,
                'confidence': min(1.0, final_confidence),
                'signal_type': consensus_type,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Meta consensus generation failed: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'signal_type': 'error',
                'reasoning': [str(e)]
            }
    
    def update_agent_performance(self, agent_name: str, predicted_action: str, actual_outcome: str, confidence: float):
        """Update agent performance tracking for adaptive weighting."""
        try:
            if agent_name not in self.agent_performance_history:
                self.agent_performance_history[agent_name] = {
                    'total_predictions': 0,
                    'correct_predictions': 0,
                    'recent_predictions': [],
                    'confidence_history': []
                }
            
            # Determine if prediction was correct
            correct = (predicted_action == actual_outcome)
            
            # Update overall statistics
            history = self.agent_performance_history[agent_name]
            history['total_predictions'] += 1
            if correct:
                history['correct_predictions'] += 1
            
            # Update recent performance (last 20 predictions)
            history['recent_predictions'].append(correct)
            history['confidence_history'].append(confidence)
            
            if len(history['recent_predictions']) > 20:
                history['recent_predictions'] = history['recent_predictions'][-20:]
                history['confidence_history'] = history['confidence_history'][-20:]
            
            # Calculate metrics
            history['historical_accuracy'] = history['correct_predictions'] / history['total_predictions']
            history['recent_accuracy'] = sum(history['recent_predictions']) / len(history['recent_predictions'])
            history['avg_confidence'] = np.mean(history['confidence_history'])
            
        except Exception as e:
            logger.error(f"Agent performance update failed: {str(e)}")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent signals and generate meta consensus."""
        try:
            if "agent_signals" not in data:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "No agent signals provided"}
                }
            
            # Generate consensus
            signal_data = self.generate_meta_consensus(data)
            
            # Get comprehensive analysis
            agent_signals = data["agent_signals"]
            market_context = data.get("market_context", {})
            
            weighted_consensus = self.weighted_voting_consensus(agent_signals)
            bayesian_consensus = self.bayesian_consensus(agent_signals)
            confidence_consensus = self.confidence_weighted_average(agent_signals)
            conflict_analysis = self.detect_signal_conflicts(agent_signals)
            
            return {
                "action": signal_data['action'],
                "confidence": signal_data['confidence'],
                "metadata": {
                    "signal_type": signal_data['signal_type'],
                    "reasoning": signal_data['reasoning'],
                    "weighted_consensus": weighted_consensus,
                    "bayesian_consensus": bayesian_consensus,
                    "confidence_consensus": confidence_consensus,
                    "conflict_analysis": conflict_analysis,
                    "agent_count": len(agent_signals)
                }
            }
            
        except Exception as e:
            logger.error(f"Meta consensus processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            } 