import logging

from src.legacy_backend_agents.risk_management.position_sizer import PositionSizer
from src.legacy_backend_agents.risk_management.risk_filter import RiskFilter


class DummyBrokerAPI:
    def place_order(self, symbol, side, quantity):
        return {"symbol": symbol, "side": side, "quantity": quantity, "status": "filled"}

class TradeExecutor:
    def __init__(self, broker_api=None, portfolio_value=100000):
        self.broker = broker_api or DummyBrokerAPI()
        self.sizer = PositionSizer(portfolio_value=portfolio_value)
        self.risk = RiskFilter()

    def execute_signal(self, signal, stop_pct):
        if not self.risk.approve(signal):
            logging.info("Trade rejected by risk filter.")
            return None
        size = self.sizer.size(stop_pct, confidence=signal.get("confidence", 0.5))
        if size <= 0:
            logging.info("Trade size zero after sizing logic.")
            return None
        order = self.broker.place_order(
            symbol=signal["symbol"],
            side="buy" if signal["signal"] == "bullish" else "sell",
            quantity=size
        )
        self.risk.track(order)
        logging.info(f"Trade executed: {order}")
        return order
