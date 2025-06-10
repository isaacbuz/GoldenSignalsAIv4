class PositionSizer:
    def __init__(self, max_risk_per_trade=0.02, portfolio_value=100000):
        self.max_risk = max_risk_per_trade
        self.portfolio = portfolio_value

    def size(self, stop_pct: float, confidence: float = 0.5) -> int:
        adjusted_risk = self.max_risk * confidence
        risk_amount = self.portfolio * adjusted_risk
        if stop_pct <= 0: return 0
        return int(risk_amount / (stop_pct * self.portfolio))
