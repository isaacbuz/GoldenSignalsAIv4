class PortfolioManager:
    def __init__(self):
        self.positions = {}

    def add_position(self, trade):
        self.positions[trade.symbol] = trade

    def get_portfolio_value(self):
        return 10000.0
