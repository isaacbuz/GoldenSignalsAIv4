from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


class AnalyticsManager:
    """Analytics functionality integrated from AlphaPy"""

    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = "arithmetic") -> pd.Series:
        """Calculate returns from price series"""
        if method == "arithmetic":
            returns = prices.pct_change()
        else:  # logarithmic
            returns = np.log(prices / prices.shift(1))
        return returns

    @staticmethod
    def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate key risk metrics"""
        metrics = {
            "volatility": returns.std() * np.sqrt(252),  # Annualized volatility
            "sharpe_ratio": (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            "max_drawdown": (returns.cumsum() - returns.cumsum().cummax()).min(),
            "skewness": stats.skew(returns.dropna()),
            "kurtosis": stats.kurtosis(returns.dropna()),
        }
        return metrics

    @staticmethod
    def calculate_portfolio_metrics(
        portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        # Calculate beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance

        # Calculate alpha (Jensen's Alpha)
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        portfolio_excess_return = portfolio_returns.mean() * 252 - risk_free_rate
        market_excess_return = benchmark_returns.mean() * 252 - risk_free_rate
        alpha = portfolio_excess_return - beta * market_excess_return

        # Information ratio
        active_returns = portfolio_returns - benchmark_returns
        information_ratio = active_returns.mean() / active_returns.std()

        metrics = {
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "tracking_error": active_returns.std() * np.sqrt(252),
        }
        return metrics

    @staticmethod
    def calculate_drawdowns(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
        """Calculate and return top drawdowns"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max

        # Find drawdown periods
        is_drawdown = drawdowns < 0
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_ends = ~is_drawdown & is_drawdown.shift(1).fillna(False)

        periods = []
        for start_date in drawdown_starts[drawdown_starts].index:
            end_dates = drawdown_ends[start_date:].index
            if len(end_dates) == 0:
                end_date = drawdowns.index[-1]
            else:
                end_date = end_dates[0]

            drawdown_value = drawdowns[start_date:end_date].min()
            duration = (end_date - start_date).days

            periods.append(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "drawdown": drawdown_value,
                    "duration": duration,
                }
            )

        df_drawdowns = pd.DataFrame(periods)
        return df_drawdowns.nlargest(top_n, "drawdown")
