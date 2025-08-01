import pytest


# Dummy domain object for illustration; replace with actual imports
class Trade:
    def __init__(self, symbol, qty):
        self.symbol = symbol
        self.qty = qty


def test_trade_creation():
    trade = Trade("AAPL", 10)
    assert trade.symbol == "AAPL"
    assert trade.qty == 10
