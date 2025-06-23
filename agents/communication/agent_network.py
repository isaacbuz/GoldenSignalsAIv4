"""
agent_network.py
Purpose: Implements agent-to-agent communication infrastructure for GoldenSignalsAI, enabling agents to exchange messages, share insights, and collaboratively improve decision-making. Uses NetworkX for communication graphs.
"""

import logging
import time
import uuid
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import networkx as nx
import numpy as np

class MessageType(Enum):
    """Types of messages agents can exchange."""
    SIGNAL = auto()
    RECOMMENDATION = auto()
    CONFLICT = auto()
    LEARNING_UPDATE = auto()

@dataclass
class AgentMessage:
    """Structured message for agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ''
    message_type: MessageType = MessageType.SIGNAL
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    confidence: float = 0.0

class AgentCommunicationNetwork:
    """
    Advanced communication network for agents to interact, 
    share insights, and collectively improve decision-making.
    """

    def __init__(self):
        """Initialize the agent communication network."""
        self.graph = nx.DiGraph()
        self.message_history = []
        self.agent_trust_scores = {}
        
    def register_agent(self, agent_name: str, agent_type: str):
        """
        Register an agent in the communication network.
        
        Args:
            agent_name (str): Unique identifier for the agent
            agent_type (str): Type of agent (e.g., 'sentiment', 'predictive')
        """
        self.graph.add_node(agent_name, type=agent_type)
        self.agent_trust_scores[agent_name] = 1.0  # Initial trust score
        
    def add_communication_channel(self, agent1: str, agent2: str, weight: float = 1.0):
        """
        Establish a communication channel between agents.
        
        Args:
            agent1 (str): First agent's name
            agent2 (str): Second agent's name
            weight (float): Communication channel strength
        """
        self.graph.add_edge(agent1, agent2, weight=weight)
        
    def broadcast_message(self, message: AgentMessage):
        """
        Broadcast a message across the agent network.
        
        Args:
            message (AgentMessage): Message to broadcast
        """
        self.message_history.append(message)
        
        # Propagate message to connected agents
        for neighbor in self.graph.neighbors(message.sender):
            # Apply trust-based message filtering
            trust_factor = self.agent_trust_scores[neighbor]
            if np.random.random() < trust_factor:
                # Simulate message transmission with potential noise
                noisy_message = self._apply_communication_noise(message)
                logging.info(f"Message from {message.sender} to {neighbor}: {noisy_message.content}")
    
    def _apply_communication_noise(self, message: AgentMessage) -> AgentMessage:
        """
        Simulate real-world communication noise and imperfection.
        
        Args:
            message (AgentMessage): Original message
        
        Returns:
            AgentMessage: Potentially modified message
        """
        noise_factor = 1 - self.agent_trust_scores[message.sender]
        
        # Introduce slight variations in confidence and content
        modified_content = message.content.copy()
        for key in modified_content:
            if isinstance(modified_content[key], (int, float)):
                modified_content[key] *= (1 + noise_factor * np.random.normal(0, 0.1))
        
        return AgentMessage(
            sender=message.sender,
            message_type=message.message_type,
            content=modified_content,
            confidence=max(0, message.confidence * (1 - noise_factor))
        )
    
    def update_trust_scores(self, agent_performance: Dict[str, float]):
        """
        Update agent trust scores based on recent performance.
        
        Args:
            agent_performance (Dict[str, float]): Performance metrics for agents
        """
        for agent, performance in agent_performance.items():
            # Exponential moving average for trust scores
            current_trust = self.agent_trust_scores.get(agent, 1.0)
            new_trust = 0.8 * current_trust + 0.2 * performance
            self.agent_trust_scores[agent] = max(0, min(new_trust, 1.0))
        
        logging.info(f"Updated trust scores: {self.agent_trust_scores}")

class CollectiveIntelligenceOrchestrator:
    """
    Orchestrates collective decision-making across the agent network.
    """
    
    def __init__(self, communication_network: AgentCommunicationNetwork):
        """
        Initialize the collective intelligence system.
        
        Args:
            communication_network (AgentCommunicationNetwork): Agent communication network
        """
        self.network = communication_network
        self.decision_history = []
    
    def aggregate_signals(self, agent_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate signals from multiple agents with weighted confidence.
        
        Args:
            agent_signals (List[Dict[str, Any]]): Signals from different agents
        
        Returns:
            Dict[str, Any]: Aggregated trading decision
        """
        # Weight signals by agent trust scores
        weighted_signals = []
        for signal in agent_signals:
            agent_name = signal.get('agent', 'unknown')
            trust_score = self.network.agent_trust_scores.get(agent_name, 0.5)
            weighted_signal = {
                **signal,
                'weighted_confidence': signal.get('confidence', 0) * trust_score
            }
            weighted_signals.append(weighted_signal)
        
        # Sort signals by weighted confidence
        sorted_signals = sorted(
            weighted_signals, 
            key=lambda x: x['weighted_confidence'], 
            reverse=True
        )
        
        # Majority voting with confidence weighting
        actions = {}
        for signal in sorted_signals:
            action = signal.get('action', 'hold')
            actions[action] = actions.get(action, 0) + signal['weighted_confidence']
        
        final_action = max(actions, key=actions.get)
        final_confidence = actions[final_action] / sum(actions.values())
        
        aggregated_decision = {
            'action': final_action,
            'confidence': final_confidence,
            'contributing_agents': [s['agent'] for s in sorted_signals[:3]]
        }
        
        self.decision_history.append(aggregated_decision)
        return aggregated_decision

# Global communication network
agent_communication_network = AgentCommunicationNetwork()
collective_intelligence = CollectiveIntelligenceOrchestrator(agent_communication_network)
