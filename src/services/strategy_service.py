from src.domain.trading.strategies.backtest_strategy import BacktestStrategy


class StrategyService:
    def __init__(self):
        self.backtest_strategy = BacktestStrategy()

    async def backtest(self, symbol, historical_df, predictions):
        result = self.backtest_strategy.run(symbol, historical_df, predictions)
        return {
            "total_return": result["total_return"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
        }
