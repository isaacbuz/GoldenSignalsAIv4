"""
Hybrid Agent Base - Supports Independent and Collaborative Signal Generation
Each agent can work alone AND collaborate with others, with performance tracking
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from collections import deque
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class HybridAgent(ABC):
    """
    Base class for agents that can generate both independent and collaborative signals
    with dynamic performance scoring
    """
    
    def __init__(self, name: str, data_bus=None):
        self.name = name
        self.data_bus = data_bus
        
        # Performance tracking
        self.independent_performance = deque(maxlen=100)  # Track last 100 signals
        self.collaborative_performance = deque(maxlen=100)
        self.divergence_wins = deque(maxlen=50)  # When going against consensus was right
        
        # Dynamic weights
        self.independent_weight = 0.5
        self.collaborative_weight = 0.5
        self.divergence_bonus = 0.0
        
        # Sentiment tracking
        self.current_sentiment = {
            'independent': None,
            'collaborative': None,
            'final': None
        }
        
    @abstractmethod
    def analyze_independent(self, symbol: str, data: Any) -> Dict[str, Any]:
        """
        Generate pure independent analysis without any external context
        Must be implemented by each agent
        """
        pass
    
    @abstractmethod
    def analyze_collaborative(self, symbol: str, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate enhanced analysis using shared context from other agents
        Must be implemented by each agent
        """
        pass
    
    def generate_signal(self, symbol: str, market_data: Any = None) -> Dict[str, Any]:
        """
        Generate both independent and collaborative signals, then blend based on performance
        """
        try:
            # 1. Generate independent signal (pure analysis)
            independent_signal = self.analyze_independent(symbol, market_data)
            self.current_sentiment['independent'] = self._extract_sentiment(independent_signal)
            
            # 2. Generate collaborative signal if data bus available
            if self.data_bus:
                context = self._get_relevant_context(symbol)
                collaborative_signal = self.analyze_collaborative(symbol, market_data, context)
                self.current_sentiment['collaborative'] = self._extract_sentiment(collaborative_signal)
                
                # Share our independent insights
                self._share_insights(symbol, independent_signal)
            else:
                collaborative_signal = independent_signal
                self.current_sentiment['collaborative'] = self.current_sentiment['independent']
            
            # 3. Detect divergence
            divergence = self._detect_divergence(independent_signal, collaborative_signal)
            
            # 4. Blend signals based on dynamic weights
            final_signal = self._blend_signals(
                independent_signal, 
                collaborative_signal, 
                divergence
            )
            
            # 5. Add performance metadata
            final_signal['metadata']['signal_components'] = {
                'independent': {
                    'action': independent_signal['action'],
                    'confidence': independent_signal['confidence'],
                    'sentiment': self.current_sentiment['independent']
                },
                'collaborative': {
                    'action': collaborative_signal['action'],
                    'confidence': collaborative_signal['confidence'],
                    'sentiment': self.current_sentiment['collaborative']
                },
                'weights': {
                    'independent': self.independent_weight,
                    'collaborative': self.collaborative_weight,
                    'divergence_bonus': self.divergence_bonus
                },
                'divergence': divergence
            }
            
            self.current_sentiment['final'] = self._extract_sentiment(final_signal)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error in {self.name} signal generation: {e}")
            return self._error_signal(str(e))
    
    def _extract_sentiment(self, signal: Dict[str, Any]) -> str:
        """Extract sentiment from signal (bullish/bearish/neutral)"""
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        
        if action == 'BUY':
            if confidence > 0.7:
                return 'strong_bullish'
            else:
                return 'bullish'
        elif action == 'SELL':
            if confidence > 0.7:
                return 'strong_bearish'
            else:
                return 'bearish'
        else:
            return 'neutral'
    
    def _detect_divergence(self, independent: Dict[str, Any], collaborative: Dict[str, Any]) -> Dict[str, Any]:
        """Detect when independent and collaborative signals diverge"""
        ind_action = independent.get('action')
        col_action = collaborative.get('action')
        
        divergence_type = 'none'
        divergence_strength = 0.0
        
        if ind_action != col_action:
            if (ind_action == 'BUY' and col_action == 'SELL') or \
               (ind_action == 'SELL' and col_action == 'BUY'):
                divergence_type = 'strong'
                divergence_strength = 1.0
            else:
                divergence_type = 'moderate'
                divergence_strength = 0.5
        
        return {
            'type': divergence_type,
            'strength': divergence_strength,
            'independent_action': ind_action,
            'collaborative_action': col_action
        }
    
    def _blend_signals(self, independent: Dict[str, Any], collaborative: Dict[str, Any], 
                      divergence: Dict[str, Any]) -> Dict[str, Any]:
        """Blend independent and collaborative signals based on performance weights"""
        
        # Calculate effective weights
        total_weight = self.independent_weight + self.collaborative_weight
        
        if divergence['type'] != 'none':
            # In case of divergence, add bonus weight to historically better performer
            if len(self.divergence_wins) > 10:
                recent_divergence_success = sum(self.divergence_wins) / len(self.divergence_wins)
                if recent_divergence_success > 0.6:  # If divergence often correct
                    self.divergence_bonus = 0.2
        
        # Normalize weights
        ind_weight = (self.independent_weight + self.divergence_bonus) / (total_weight + self.divergence_bonus)
        col_weight = self.collaborative_weight / (total_weight + self.divergence_bonus)
        
        # Blend confidences
        ind_conf = independent.get('confidence', 0)
        col_conf = collaborative.get('confidence', 0)
        blended_confidence = (ind_conf * ind_weight) + (col_conf * col_weight)
        
        # Determine action (weighted voting)
        action_scores = {}
        for action in ['BUY', 'SELL', 'HOLD']:
            ind_score = ind_conf if independent.get('action') == action else 0
            col_score = col_conf if collaborative.get('action') == action else 0
            action_scores[action] = (ind_score * ind_weight) + (col_score * col_weight)
        
        final_action = max(action_scores.items(), key=lambda x: x[1])[0]
        
        # Build reasoning
        reasoning_parts = []
        if independent['action'] == collaborative['action']:
            reasoning_parts.append(f"Consensus {final_action}")
        else:
            reasoning_parts.append(f"Mixed signals: Ind={independent['action']}, Col={collaborative['action']}")
        
        reasoning_parts.append(f"Confidence: {blended_confidence:.2f}")
        reasoning_parts.append(f"Weights: Ind={ind_weight:.2f}, Col={col_weight:.2f}")
        
        if divergence['type'] != 'none':
            reasoning_parts.append(f"Divergence: {divergence['type']}")
        
        return {
            'action': final_action,
            'confidence': blended_confidence,
            'metadata': {
                'agent': self.name,
                'reasoning': ' | '.join(reasoning_parts),
                'timestamp': datetime.now().isoformat(),
                'blend_method': 'dynamic_weighted'
            }
        }
    
    def update_performance(self, signal_id: str, outcome: float, signal_type: str = 'final'):
        """
        Update performance metrics based on signal outcome
        outcome: 1.0 = correct, -1.0 = wrong, 0 = neutral
        """
        if signal_type == 'independent':
            self.independent_performance.append(outcome)
        elif signal_type == 'collaborative':
            self.collaborative_performance.append(outcome)
        elif signal_type == 'divergence':
            self.divergence_wins.append(outcome > 0)
        
        # Update dynamic weights based on performance
        self._update_weights()
    
    def _update_weights(self):
        """Update dynamic weights based on recent performance"""
        if len(self.independent_performance) >= 20 and len(self.collaborative_performance) >= 20:
            # Calculate recent performance
            ind_score = np.mean(list(self.independent_performance)[-20:])
            col_score = np.mean(list(self.collaborative_performance)[-20:])
            
            # Adjust weights proportionally
            total = abs(ind_score) + abs(col_score)
            if total > 0:
                self.independent_weight = 0.3 + (0.4 * (ind_score + 1) / 2)  # Range: 0.3-0.7
                self.collaborative_weight = 0.3 + (0.4 * (col_score + 1) / 2)
                
                # Ensure weights are normalized
                weight_sum = self.independent_weight + self.collaborative_weight
                self.independent_weight /= weight_sum
                self.collaborative_weight /= weight_sum
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        def calculate_metrics(performance_list):
            if not performance_list:
                return {'accuracy': 0.5, 'avg_outcome': 0, 'total_signals': 0}
            
            outcomes = list(performance_list)
            correct = sum(1 for o in outcomes if o > 0)
            total = len(outcomes)
            
            return {
                'accuracy': correct / total if total > 0 else 0.5,
                'avg_outcome': np.mean(outcomes),
                'total_signals': total,
                'recent_trend': np.mean(outcomes[-10:]) if len(outcomes) >= 10 else 0
            }
        
        return {
            'agent': self.name,
            'independent': calculate_metrics(self.independent_performance),
            'collaborative': calculate_metrics(self.collaborative_performance),
            'divergence_success_rate': sum(self.divergence_wins) / len(self.divergence_wins) if self.divergence_wins else 0.5,
            'current_weights': {
                'independent': self.independent_weight,
                'collaborative': self.collaborative_weight,
                'divergence_bonus': self.divergence_bonus
            },
            'current_sentiment': self.current_sentiment
        }
    
    def _get_relevant_context(self, symbol: str) -> Dict[str, Any]:
        """Get relevant context for this specific agent type"""
        # Override in subclasses to specify what context each agent needs
        if self.data_bus:
            return self.data_bus.get_context(symbol)
        return {}
    
    def _share_insights(self, symbol: str, signal: Dict[str, Any]):
        """Share this agent's insights with others"""
        # Override in subclasses to specify what to share
        pass
    
    def _error_signal(self, error_msg: str) -> Dict[str, Any]:
        """Generate error signal"""
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'metadata': {
                'agent': self.name,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
        }


# Example: Sentiment aggregation across all agents
class SentimentAggregator:
    """Aggregates sentiment from all agents to create market-wide sentiment"""
    
    def __init__(self):
        self.agent_sentiments = {}
        
    def update_sentiment(self, agent_name: str, sentiment_data: Dict[str, str]):
        """Update sentiment from an agent"""
        self.agent_sentiments[agent_name] = {
            'independent': sentiment_data.get('independent'),
            'collaborative': sentiment_data.get('collaborative'),
            'final': sentiment_data.get('final'),
            'timestamp': datetime.now()
        }
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Calculate overall market sentiment from all agents"""
        if not self.agent_sentiments:
            return {'overall': 'neutral', 'confidence': 0}
        
        # Count sentiments
        sentiment_counts = {
            'bullish': 0, 'strong_bullish': 0,
            'bearish': 0, 'strong_bearish': 0,
            'neutral': 0
        }
        
        # Weight by recency
        for agent, data in self.agent_sentiments.items():
            age = (datetime.now() - data['timestamp']).total_seconds()
            if age < 300:  # Only count recent sentiments (5 min)
                weight = 1.0 - (age / 300) * 0.5  # Decay weight over time
                
                final_sentiment = data.get('final', 'neutral')
                if final_sentiment in sentiment_counts:
                    sentiment_counts[final_sentiment] += weight
        
        # Determine overall sentiment
        total = sum(sentiment_counts.values())
        if total == 0:
            return {'overall': 'neutral', 'confidence': 0}
        
        bull_score = sentiment_counts['bullish'] + sentiment_counts['strong_bullish'] * 1.5
        bear_score = sentiment_counts['bearish'] + sentiment_counts['strong_bearish'] * 1.5
        
        if bull_score > bear_score * 1.5:
            overall = 'bullish'
            confidence = bull_score / total
        elif bear_score > bull_score * 1.5:
            overall = 'bearish'
            confidence = bear_score / total
        else:
            overall = 'neutral'
            confidence = sentiment_counts['neutral'] / total
        
        return {
            'overall': overall,
            'confidence': confidence,
            'breakdown': sentiment_counts,
            'agent_count': len(self.agent_sentiments)
        } 