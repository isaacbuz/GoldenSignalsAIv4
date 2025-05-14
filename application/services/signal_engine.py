# application/services/signal_engine.py
# Purpose: Manage signal generation and processing across multiple agents

import logging
from typing import Dict, Any, List

from agents.factory import AgentFactory

# Configure logging
logger = logging.getLogger(__name__)

class SignalEngine:
    """
    Manages signal generation and processing across multiple agents.
    """

    def __init__(self, agent_factory: AgentFactory):
        """
        Initialize the SignalEngine with an AgentFactory.

        Args:
            agent_factory (AgentFactory): Factory for creating and managing agents.
        """
        self.agent_factory = agent_factory
        self.agents = {}
        logger.info("SignalEngine initialized")

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals from multiple agents.

        Args:
            market_data (Dict[str, Any]): Comprehensive market data.

        Returns:
            List[Dict[str, Any]]: List of generated trading signals.
        """
        signals = []
        try:
            # Get all available agents
            for agent_type in self.agent_factory.get_available_agents():
                agent = self.agent_factory.create_agent(agent_type)
                
                # Process market data through the agent
                signal = agent.process(market_data)
                
                # Optional: Process the signal further
                processed_signal = agent.process_signal(signal)
                
                signals.append(processed_signal)
                
            logger.info(f"Generated {len(signals)} signals")
            return signals
        
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            return []

    def evaluate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate and aggregate signals to make a final trading decision.

        Args:
            signals (List[Dict[str, Any]]): List of trading signals.

        Returns:
            Dict[str, Any]: Aggregated trading decision.
        """
        try:
            # Simple aggregation strategy
            buy_signals = [s for s in signals if s.get('action') == 'buy']
            sell_signals = [s for s in signals if s.get('action') == 'sell']
            
            # Weighted confidence calculation
            total_buy_confidence = sum(s.get('confidence', 0) for s in buy_signals)
            total_sell_confidence = sum(s.get('confidence', 0) for s in sell_signals)
            
            if total_buy_confidence > total_sell_confidence:
                final_decision = {
                    'action': 'buy',
                    'confidence': total_buy_confidence / len(buy_signals) if buy_signals else 0.0,
                    'metadata': {
                        'buy_signals': len(buy_signals),
                        'sell_signals': len(sell_signals)
                    }
                }
            elif total_sell_confidence > total_buy_confidence:
                final_decision = {
                    'action': 'sell',
                    'confidence': total_sell_confidence / len(sell_signals) if sell_signals else 0.0,
                    'metadata': {
                        'buy_signals': len(buy_signals),
                        'sell_signals': len(sell_signals)
                    }
                }
            else:
                final_decision = {
                    'action': 'hold',
                    'confidence': 0.0,
                    'metadata': {
                        'buy_signals': len(buy_signals),
                        'sell_signals': len(sell_signals)
                    }
                }
            
            logger.info(f"Final trading decision: {final_decision}")
            return final_decision
        
        except Exception as e:
            logger.error(f"Signal evaluation failed: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'metadata': {}}
