from backend.agents.base import BaseSignalAgent
import numpy as np

class BacktestAgent(BaseSignalAgent):
    """
    Runs backtests on historical data for a given agent and reports performance metrics.
    """
    def __init__(self, symbol: str, test_agent):
        super().__init__(symbol)
        self.test_agent = test_agent

    def run(self, price_history: list) -> dict:
        # Placeholder: simple strategy using test_agent's signal
        returns = []
        for i in range(10, len(price_history)):
            window = price_history[i-10:i]
            signal = self.test_agent.run(window).get('signal', 'hold')
            ret = np.log(price_history[i] / price_history[i-1])
            if signal == 'buy':
                returns.append(ret)
            elif signal == 'sell':
                returns.append(-ret)
            else:
                returns.append(0)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        explanation = f"Backtest Sharpe ratio: {sharpe:.2f}"
        return {"agent": "BacktestAgent", "sharpe": float(sharpe), "explanation": explanation}
