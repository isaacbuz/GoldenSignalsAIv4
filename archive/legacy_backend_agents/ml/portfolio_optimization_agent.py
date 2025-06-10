import numpy as np
from scipy.optimize import minimize
from archive.legacy_backend_agents.base import BaseSignalAgent

class PortfolioOptimizationAgent(BaseSignalAgent):
    """
    Suggests optimal portfolio weights for a list of assets using mean-variance optimization.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)

    def run(self, returns_matrix: np.ndarray) -> dict:
        # returns_matrix: shape (n_days, n_assets)
        n = returns_matrix.shape[1]
        mean_returns = np.mean(returns_matrix, axis=0)
        cov = np.cov(returns_matrix, rowvar=False)
        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            return -port_return / (port_vol + 1e-8)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        res = minimize(neg_sharpe, np.ones(n)/n, bounds=bounds, constraints=cons)
        explanation = f"Optimal weights: {res.x.round(2)}"
        return {"agent": "PortfolioOptimizationAgent", "weights": res.x.tolist(), "confidence": 85, "explanation": explanation}
