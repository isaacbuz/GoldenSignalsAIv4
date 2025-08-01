from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from strategies.advanced_strategies import AdvancedTradingStrategies


class StrategyOrchestrator:
    """
    Centralized orchestrator for managing and executing trading strategies.
    """

    def __init__(self,
                 strategies: Optional[List[str]] = None,
                 strategy_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the strategy orchestrator.

        Args:
            strategies (List[str], optional): List of strategies to use
            strategy_weights (Dict[str, float], optional): Custom weights for strategies
        """
        # Default strategies if not specified
        self.available_strategies = {
            'pairs_trading': {
                'method': AdvancedTradingStrategies.pairs_trading,
                'default_weight': 0.2
            },
            'momentum': {
                'method': AdvancedTradingStrategies.momentum_strategy,
                'default_weight': 0.25
            },
            'volatility_breakout': {
                'method': AdvancedTradingStrategies.volatility_breakout,
                'default_weight': 0.2
            },
            'pattern_recognition': {
                'method': AdvancedTradingStrategies.pattern_recognition,
                'default_weight': 0.15
            },
            'adaptive': {
                'method': AdvancedTradingStrategies.adaptive_strategy,
                'default_weight': 0.2
            }
        }

        # Select strategies
        self.active_strategies = strategies or list(self.available_strategies.keys())

        # Set strategy weights
        self.strategy_weights = strategy_weights or {
            strategy: self.available_strategies[strategy]['default_weight']
            for strategy in self.active_strategies
        }

        # Performance tracking
        self.strategy_performance = {
            strategy: {'wins': 0, 'total_trades': 0}
            for strategy in self.active_strategies
        }

    def execute_strategies(self, market_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Execute multiple trading strategies on market data.

        Args:
            market_data (Dict[str, np.ndarray]): Market data for strategy execution

        Returns:
            Dict[str, Any]: Comprehensive strategy execution results
        """
        # Validate market data
        required_keys = ['prices', 'high', 'low', 'close']
        for key in required_keys:
            if key not in market_data:
                raise ValueError(f"Missing required market data: {key}")

        # Execute strategies
        strategy_results = {}
        weighted_signals = np.zeros_like(market_data['prices'])

        for strategy_name in self.active_strategies:
            strategy_method = self.available_strategies[strategy_name]['method']

            try:
                # Strategy-specific execution
                if strategy_name == 'pairs_trading':
                    result = strategy_method(
                        market_data['prices'],
                        market_data.get('paired_asset_prices', market_data['prices'])
                    )
                elif strategy_name in ['momentum', 'volatility_breakout', 'adaptive']:
                    result = strategy_method(market_data['prices'])
                elif strategy_name == 'pattern_recognition':
                    result = strategy_method(market_data['prices'])
                else:
                    continue

                # Weight signals
                weight = self.strategy_weights.get(strategy_name, 0.1)
                strategy_signals = result['signals'] * weight
                weighted_signals += strategy_signals

                # Track strategy results
                strategy_results[strategy_name] = result

            except Exception as e:
                print(f"Error in {strategy_name} strategy: {e}")

        # Final signal aggregation
        final_signals = np.sign(weighted_signals)

        return {
            'final_signals': final_signals,
            'strategy_results': strategy_results,
            'strategy_weights': self.strategy_weights
        }

    def update_strategy_weights(self, performance_metrics: Dict[str, float]):
        """
        Dynamically update strategy weights based on performance.

        Args:
            performance_metrics (Dict[str, float]): Performance scores for strategies
        """
        total_performance = sum(performance_metrics.values())

        for strategy, performance in performance_metrics.items():
            # Proportional weight adjustment
            if total_performance > 0:
                self.strategy_weights[strategy] *= (1 + performance / total_performance)

        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        self.strategy_weights = {
            k: v / total_weight
            for k, v in self.strategy_weights.items()
        }

    def backtest_strategies(
        self,
        historical_data: Dict[str, np.ndarray],
        initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """
        Comprehensive backtesting of strategies.

        Args:
            historical_data (Dict[str, np.ndarray]): Historical market data
            initial_capital (float): Starting capital for simulation

        Returns:
            Dict[str, Any]: Backtesting results
        """
        capital = initial_capital
        portfolio_value = [initial_capital]

        for i in range(1, len(historical_data['prices'])):
            # Execute strategies
            strategy_results = self.execute_strategies({
                key: data[:i] for key, data in historical_data.items()
            })

            # Trading decision
            signal = strategy_results['final_signals'][-1]
            price = historical_data['prices'][i]

            # Simple trading simulation
            if signal > 0:  # Buy signal
                shares = capital // price
                capital -= shares * price
            elif signal < 0:  # Sell signal
                capital += shares * price
                shares = 0

            # Track portfolio value
            portfolio_value.append(capital + shares * price)

        return {
            'final_portfolio_value': portfolio_value[-1],
            'total_return': (portfolio_value[-1] - initial_capital) / initial_capital * 100,
            'portfolio_history': portfolio_value
        }

def main():
    """
    Demonstration of strategy orchestrator.
    """
    # Simulate market data
    np.random.seed(42)
    market_data = {
        'prices': np.cumsum(np.random.normal(0, 1, 1000)),
        'high': np.cumsum(np.random.normal(0, 1, 1000)),
        'low': np.cumsum(np.random.normal(0, 1, 1000)),
        'close': np.cumsum(np.random.normal(0, 1, 1000))
    }

    # Initialize orchestrator
    orchestrator = StrategyOrchestrator()

    # Execute strategies
    results = orchestrator.execute_strategies(market_data)
    print("Strategy Results:", results)

    # Backtest
    backtest_results = orchestrator.backtest_strategies(market_data)
    print("Backtest Results:", backtest_results)

if __name__ == '__main__':
    main()
