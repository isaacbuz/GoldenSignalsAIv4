import numpy as np
import pandas as pd
import yaml
from strategies.advanced_strategies import AdvancedStrategies

from src.infrastructure.error_handler import DataFetchError, ErrorHandler, ModelInferenceError
from src.ml.models.factory import ModelFactory


class BacktestResult:
    def __init__(self, equity, metrics):
        self.equity = equity
        self.metrics = metrics

class BacktestEngine:
    """
    Unified backtest engine for GoldenSignalsAI. Loads all parameters from config/parameters.yaml, uses strategy/model registries, and advanced error handling.
    """
    def __init__(self, price_df, signal_series, config_path='config/parameters.yaml'):
        self.price = price_df['close']
        self.signal = signal_series
        self.initial_capital = 10000
        self.commission = 0.001
        self.slippage = 0.0005
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.strategy_params = self.config.get('strategies', {})
        self.model_params = self.config.get('models', {})

    def run(self) -> BacktestResult:
        """
        Runs the backtest using unified logic, config-driven parameters, and advanced error handling.
        Returns:
            BacktestResult: Results of the backtest
        """
        try:
            positions = self.signal.shift().fillna(0)  # 1/-1/0 positions
            returns = self.price.pct_change().fillna(0)
            # Apply slippage and commission
            net_returns = positions * returns - np.abs(positions.diff()) * self.commission
            net_returns -= np.abs(positions) * self.slippage * returns.abs()
            equity = (1 + net_returns).cumprod() * self.initial_capital
            # Compute metrics
            total_return = equity.iloc[-1] / self.initial_capital - 1
            sharpe_ratio = net_returns.mean() / net_returns.std() * (252 ** 0.5) if net_returns.std() != 0 else 0
            max_drawdown = (equity - equity.cummax()).min()
            # Win/loss stats
            wins = (net_returns > 0).sum()
            losses = (net_returns < 0).sum()
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            avg_trade_return = net_returns.mean()
            metrics = {
                "total_return": float(total_return),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "avg_trade_return": float(avg_trade_return),
                "num_trades": int(len(net_returns)),
                "slippage": float(self.slippage),
                "commission": float(self.commission),
                "position_size": float(1.0)
            }
            return BacktestResult(equity=equity, metrics=metrics)
        except Exception as e:
            ErrorHandler.handle_error(e)
            raise

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('data.csv')
    # Create signal series
    signal = pd.Series([1, -1, 0, 1, -1, 0])
    # Create backtest engine
    engine = BacktestEngine(data, signal)
    # Run backtest
    result = engine.run()
    # Print results
    print(result.metrics)
