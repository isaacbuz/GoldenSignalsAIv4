"""
BacktestResearchAgent for strategy optimization via backtesting.
"""

import logging
import pandas as pd
from typing import Dict, Any
from src.services.backtest import Backtester
from agents.base import BaseAgent

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

class BacktestResearchAgent(BaseAgent):
    """Agent that researches optimal trading strategies through backtesting."""
    
    def __init__(self, max_strategies: int = 10):
        """Initialize the BacktestResearchAgent.
        
        Args:
            max_strategies (int): Maximum number of strategies to test.
        """
        super().__init__(name="BacktestResearch", agent_type="research")
        self.max_strategies = max_strategies
        self.backtester = Backtester()
        self.tested_strategies = []
        self.results = []
        logger.info({
            "message": f"BacktestResearchAgent initialized with max_strategies={max_strategies}"
        })

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process historical data to run backtests on multiple strategies.
        
        Args:
            data (Dict): Market observation with 'stock_data', 'signals'.
        
        Returns:
            Dict: Decision with best strategy and metadata.
        """
        logger.info({"message": "Processing data for BacktestResearchAgent"})
        try:
            stock_data = pd.DataFrame(data['stock_data'])
            signals = pd.DataFrame(data.get('signals', []))
            if stock_data.empty or signals.empty:
                logger.warning({"message": "No stock data or signals available"})
                return {"best_strategy": None, "performance": {}, "metadata": {}}

            # Run backtest with mock variations (simplified)
            backtester = Backtester(stock_data, signals)
            result = backtester.run(initial_capital=10000)
            if "error" in result:
                logger.error({"message": f"Backtest failed: {result['error']}"})
                return {"best_strategy": None, "performance": {}, "metadata": {"error": result['error']}}

            self.results.append(result)
            if len(self.results) > self.max_strategies:
                self.results.pop(0)

            # Select best strategy (highest Sharpe ratio)
            best_result = max(self.results, key=lambda x: x['sharpe_ratio'])
            decision = {
                "best_strategy": "mock_strategy",  # Placeholder
                "performance": {
                    "sharpe_ratio": best_result['sharpe_ratio'],
                    "total_return": best_result['total_return']
                },
                "metadata": {"num_strategies_tested": len(self.results)}
            }
            logger.info({"message": f"BacktestResearchAgent decision: {decision}"})
            return decision
        except Exception as e:
            logger.error({"message": f"BacktestResearchAgent processing failed: {str(e)}"})
            return {"best_strategy": None, "performance": {}, "metadata": {"error": str(e)}}

    def adapt(self, new_data: pd.DataFrame):
        """Adapt the agent to new market data (placeholder for learning).
        
        Args:
            new_data (pd.DataFrame): New market data.
        """
        pass 