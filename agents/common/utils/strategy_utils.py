"""
strategy_utils.py

Provides strategy optimization and tuning utilities for GoldenSignalsAI agents.
Includes:
- StrategyTuner: Hyperparameter optimizer for signal engine weights using Optuna.

Migrated from application/strategies for unified access by agents and research modules.
"""
import optuna
from goldensignalsai.application.services.signal_engine import SignalEngine

class StrategyTuner:
    """Hyperparameter optimizer for signal engine weights using Optuna."""
    def __init__(self, data, symbol, historical_returns):
        self.data = data
        self.symbol = symbol
        self.historical_returns = historical_returns
    def objective(self, trial):
        weights = {
            "ma_cross": trial.suggest_float("ma_cross", 0.1, 0.3),
            "ema_cross": trial.suggest_float("ema_cross", 0.1, 0.3),
            "vwap": trial.suggest_float("vwap", 0.1, 0.3),
            "bollinger": trial.suggest_float("bollinger", 0.1, 0.3),
            "rsi": trial.suggest_float("rsi", 0.1, 0.3),
            "macd": trial.suggest_float("macd", 0.1, 0.3)
        }
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        signal_engine = SignalEngine(self.data, weights)
        signals = []
        for i in range(len(self.data) - 1):
            temp_df = self.data.iloc[:i+1]
            signal_engine.data = temp_df
            signal = signal_engine.generate_signal(self.symbol)
            signals.append(signal)
        returns = []
        position = 0
        for i, signal in enumerate(signals):
            if signal["action"] == "Buy" and position == 0:
                position = 1
                entry_price = signal["price"]
            elif signal["action"] == "Sell" and position == 1:
                position = 0
                exit_price = signal["price"]
                trade_return = (exit_price - entry_price) / entry_price
                returns.append(trade_return)
        cumulative_return = sum(returns) if returns else 0
        return cumulative_return
    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        best_weights = study.best_params
        total = sum(best_weights.values())
        best_weights = {k: v / total for k, v in best_weights.items()}
        return best_weights
