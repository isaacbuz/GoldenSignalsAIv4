import pandas as pd
from typing import Dict, List

class PortfolioSimulator:
    def __init__(self, allocation: Dict[str, float]):
        self.allocation = allocation  # e.g. {"AAPL": 0.3, "TSLA": 0.7}
        self.portfolio_history = []

    def simulate(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # price_data: {"AAPL": df, "TSLA": df}
        returns = pd.DataFrame()
        for symbol, df in price_data.items():
            df = df.set_index("date")["close"].pct_change().fillna(0)
            returns[symbol] = df * self.allocation.get(symbol, 0)

        returns["portfolio"] = returns.sum(axis=1)
        cumulative = (1 + returns["portfolio"]).cumprod()
        self.portfolio_history = cumulative
        return cumulative

    def performance_metrics(self) -> Dict:
        if not isinstance(self.portfolio_history, pd.Series):
            return {}

        returns = self.portfolio_history.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * (252 ** 0.5)
        max_dd = (self.portfolio_history / self.portfolio_history.cummax() - 1).min()

        return {
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "final_return": round(self.portfolio_history.iloc[-1] - 1, 2)
        }
