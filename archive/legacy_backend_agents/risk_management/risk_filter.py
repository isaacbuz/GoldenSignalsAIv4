class RiskFilter:
    def __init__(self, max_open_trades: int = 5, max_drawdown: float = 0.1):
        self.max_trades = max_open_trades
        self.max_dd = max_drawdown
        self.open_trades = []
        self.drawdown = 0.0

    def approve(self, new_signal: dict) -> bool:
        if len(self.open_trades) >= self.max_trades:
            return False
        if self.drawdown > self.max_dd:
            return False
        return True

    def update_drawdown(self, current_value: float, peak: float):
        self.drawdown = max(0, (peak - current_value) / peak)

    def track(self, trade: dict):
        self.open_trades.append(trade)
