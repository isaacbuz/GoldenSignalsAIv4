import pandas as pd

class BacktestEngine:
    def run(self, data, params):
        returns = pd.Series(data['close']).pct_change().fillna(0)
        total_return = returns.sum()
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() != 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown)
        }
