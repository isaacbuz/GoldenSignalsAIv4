"""
Strategy tuning utilities for optimizing trading strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Callable
from scipy.optimize import minimize
from ..monitoring.monitoring_agents import AgentPerformanceTracker

logger = logging.getLogger(__name__)

class StrategyTuner:
    """Utility for tuning and optimizing trading strategies."""
    
    def __init__(self, performance_tracker: AgentPerformanceTracker = None):
        """Initialize the strategy tuner.
        
        Args:
            performance_tracker (AgentPerformanceTracker, optional): Performance tracking utility.
        """
        self.performance_tracker = performance_tracker or AgentPerformanceTracker()
        self.strategy_weights = {}
        self.optimization_history = []
        
    def optimize_weights(
        self,
        strategy_returns: Dict[str, pd.Series],
        target_volatility: float = 0.15,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Dict[str, float]:
        """Optimize strategy weights using mean-variance optimization.
        
        Args:
            strategy_returns (Dict[str, pd.Series]): Historical returns for each strategy.
            target_volatility (float): Target portfolio volatility.
            min_weight (float): Minimum weight per strategy.
            max_weight (float): Maximum weight per strategy.
            
        Returns:
            Dict[str, float]: Optimized strategy weights.
        """
        try:
            # Convert returns to DataFrame
            returns_df = pd.DataFrame(strategy_returns)
            
            # Calculate mean returns and covariance
            mean_returns = returns_df.mean()
            cov_matrix = returns_df.cov()
            
            # Define optimization objective (maximize Sharpe ratio)
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -portfolio_return / portfolio_vol
            
            # Constraints
            n_assets = len(strategy_returns)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))) - target_volatility}  # target volatility
            ]
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            
            # Initial guess (equal weights)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Store results
            optimized_weights = {
                strategy: weight
                for strategy, weight in zip(strategy_returns.keys(), result.x)
            }
            self.strategy_weights = optimized_weights
            self.optimization_history.append({
                'weights': optimized_weights,
                'objective_value': result.fun,
                'success': result.success
            })
            
            logger.info({
                "message": "Strategy weights optimized",
                "weights": optimized_weights
            })
            return optimized_weights
            
        except Exception as e:
            logger.error({"message": f"Strategy weight optimization failed: {str(e)}"})
            return {strategy: 1/len(strategy_returns) for strategy in strategy_returns}
            
    def evaluate_strategy(
        self,
        strategy_fn: Callable,
        test_data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a strategy with given parameters on test data.
        
        Args:
            strategy_fn (Callable): Strategy function to evaluate.
            test_data (pd.DataFrame): Historical data for testing.
            parameters (Dict[str, Any]): Strategy parameters.
            
        Returns:
            Dict[str, Any]: Strategy performance metrics.
        """
        try:
            # Run strategy
            signals = strategy_fn(test_data, parameters)
            returns = pd.Series(index=test_data.index)
            
            # Calculate returns (simplified)
            for i in range(1, len(signals)):
                if signals[i-1] == 1:  # Long
                    returns.iloc[i] = (test_data['Close'].iloc[i] / test_data['Close'].iloc[i-1]) - 1
                elif signals[i-1] == -1:  # Short
                    returns.iloc[i] = 1 - (test_data['Close'].iloc[i] / test_data['Close'].iloc[i-1])
                else:
                    returns.iloc[i] = 0
                    
            # Calculate metrics
            total_return = (1 + returns).prod() - 1
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
            
            metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': (returns > 0).mean()
            }
            
            logger.info({
                "message": "Strategy evaluation completed",
                "metrics": metrics
            })
            return metrics
            
        except Exception as e:
            logger.error({"message": f"Strategy evaluation failed: {str(e)}"})
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
            
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the history of optimization results.
        
        Returns:
            List[Dict[str, Any]]: History of optimization results.
        """
        return self.optimization_history 