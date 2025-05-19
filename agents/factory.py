"""
factory.py
Purpose: Defines the AgentFactory class for creating and orchestrating multiple agent types and trading strategies in GoldenSignalsAI. Integrates sentiment, predictive, and risk agents as well as strategy orchestration.
"""
import logging
import numpy as np
from typing import Dict, Any, List, Optional

from agents.sentiment.social_media import SocialMediaSentimentAgent
from agents.sentiment.news import NewsSentimentAgent
from agents.predictive.options_chain import OptionsChainAgent
from agents.predictive.options_flow import OptionsFlowAgent
from agents.predictive.reversion import ReversionAgent
from agents.predictive.momentum_divergence import MomentumDivergenceAgent
from agents.risk.options_risk import OptionsRiskAgent
from strategies.strategy_orchestrator import StrategyOrchestrator
from strategies.advanced_strategies import AdvancedTradingStrategies

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Advanced agent factory with strategy orchestration and multi-agent signal processing.
    Handles instantiation of agents, strategy orchestration, and multi-agent workflows.
    """

    def __init__(
        self, 
        data_fetcher=None, 
        historical_data: Optional[np.ndarray] = None,
        strategies: Optional[List[str]] = None
    ):
        """
        Initialize the agent factory with advanced capabilities.

        Args:
            data_fetcher (DataFetcher, optional): Data fetching service
            historical_data (np.ndarray, optional): Historical market data
            strategies (List[str], optional): Specific strategies to activate
        """
        self.data_fetcher = data_fetcher
        self.agents = {}
        
        # Initialize strategy orchestrator for multi-strategy workflows
        self.strategy_orchestrator = StrategyOrchestrator(
            strategies=strategies, 
            strategy_weights=None
        )
        
        # Optional historical data for strategy learning
        self.historical_data = historical_data

    def create_agents(self) -> Dict[str, Any]:
        """
        Create and configure all trading agents with advanced capabilities.

        Returns:
            Dict[str, Any]: Configured agents
        """
        self.agents = {
            'social_media_sentiment': SocialMediaSentimentAgent(),
            'news_sentiment': NewsSentimentAgent(),
            'options_chain': OptionsChainAgent(),
            'options_flow': OptionsFlowAgent(),
            'reversion': ReversionAgent(),
            'options_risk': OptionsRiskAgent(),
            'momentum_divergence': MomentumDivergenceAgent()
        }

        return self.agents

    def get_agent(self, agent_name: str) -> Any:
        """
        Retrieve a specific agent.

        Args:
            agent_name (str): Name of the agent

        Returns:
            Any: Requested agent
        """
        return self.agents.get(agent_name)

    def process_signals(self, market_data: Dict) -> Dict[str, Any]:
        """
        Advanced signal processing with multi-agent and strategy integration.

        Args:
            market_data (Dict): Comprehensive market data

        Returns:
            Dict[str, Any]: Comprehensive trading analysis
        """
        # Prepare market data for strategy processing
        strategy_market_data = {
            'prices': market_data.get('prices', []),
            'high': market_data.get('high', []),
            'low': market_data.get('low', []),
            'close': market_data.get('close', [])
        }

        # Process agent-specific signals
        agent_signals = []
        for name, agent in self.agents.items():
            try:
                signal = agent.process(market_data)
                signal['agent'] = name
                agent_signals.append(signal)
            except Exception as e:
                logger.error(f"Error processing signal from {name}: {e}")

        # Execute advanced trading strategies
        try:
            strategy_results = self.strategy_orchestrator.execute_strategies(strategy_market_data)
        except Exception as e:
            logger.error(f"Strategy execution error: {e}")
            strategy_results = {'final_signals': np.zeros(len(strategy_market_data['prices'])), 'strategy_results': {}}

        # Combine agent signals with strategy signals
        combined_analysis = {
            'agent_signals': agent_signals,
            'strategy_results': strategy_results,
            'final_trading_signal': strategy_results.get('final_signals', [])[-1] if strategy_results.get('final_signals') else 0
        }

        # Optional: Periodic strategy weight update and backtesting
        if self.historical_data is not None:
            try:
                backtest_results = self.strategy_orchestrator.backtest_strategies(
                    self.historical_data
                )
                combined_analysis['backtest_results'] = backtest_results
            except Exception as e:
                logger.error(f"Backtesting error: {e}")

        return combined_analysis

    def optimize_strategies(self, performance_metrics: Dict[str, float]):
        """
        Optimize strategy weights based on performance.

        Args:
            performance_metrics (Dict[str, float]): Performance scores for strategies
        """
        self.strategy_orchestrator.update_strategy_weights(performance_metrics)

def main():
    """
    Demonstration of enhanced agent factory.
    """
    # Simulate market data
    np.random.seed(42)
    market_data = {
        'prices': np.cumsum(np.random.normal(0, 1, 1000)),
        'high': np.cumsum(np.random.normal(0, 1, 1000)),
        'low': np.cumsum(np.random.normal(0, 1, 1000)),
        'close': np.cumsum(np.random.normal(0, 1, 1000))
    }
    
    # Initialize factory
    factory = AgentFactory(historical_data=market_data)
    factory.create_agents()
    
    # Process signals
    results = factory.process_signals(market_data)
    print("Trading Analysis:", results)

if __name__ == '__main__':
    main()
