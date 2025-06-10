"""
backtesting_utils.py
Provides utilities for backtesting trading agents and strategies on historical data.
"""
from typing import List, Dict, Any, Callable
import numpy as np

class BacktestingUtils:
    @staticmethod
    def run_backtest(agent, historical_data: List[Dict[str, Any]], signal_key: str = 'trend') -> Dict[str, Any]:
        """
        Run a backtest for a given agent on historical data.
        Args:
            agent: The agent to test (must have a run() method).
            historical_data: List of dicts with market data.
            signal_key: The key in the agent's output to use as the trading signal.
        Returns:
            Dict with statistics (accuracy, returns, etc.).
        """
        signals = []
        actuals = []
        for data in historical_data:
            result = agent.run(data)
            signals.append(result.get(signal_key))
            actuals.append(data.get('actual'))  # Expected to be in data for evaluation
        accuracy = BacktestingUtils.calculate_accuracy(signals, actuals)
        return {
            'signals': signals,
            'actuals': actuals,
            'accuracy': accuracy
        }

    @staticmethod
    def calculate_accuracy(signals, actuals):
        correct = sum(1 for s, a in zip(signals, actuals) if s == a)
        return correct / len(signals) if signals else 0
