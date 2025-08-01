from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_date: datetime


class PortfolioManager:
    """Portfolio management functionality integrated from AlphaPy"""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades_history: List[Dict] = []

    def place_order(
        self, symbol: str, quantity: float, price: float, order_type: str = "market"
    ) -> bool:
        """Place a new order"""
        order_value = quantity * price

        if quantity > 0 and order_value > self.cash:
            return False

        if quantity < 0 and (
            symbol not in self.positions or abs(quantity) > self.positions[symbol].quantity
        ):
            return False

        # Update cash
        self.cash -= order_value

        # Update position
        if symbol in self.positions:
            current_position = self.positions[symbol]
            new_quantity = current_position.quantity + quantity

            if new_quantity == 0:
                del self.positions[symbol]
            else:
                # Calculate new average entry price for buys
                if quantity > 0:
                    total_value = (
                        current_position.quantity * current_position.entry_price + quantity * price
                    )
                    new_entry_price = total_value / new_quantity
                else:
                    new_entry_price = current_position.entry_price

                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=new_quantity,
                    entry_price=new_entry_price,
                    entry_date=datetime.now() if quantity > 0 else current_position.entry_date,
                )
        else:
            self.positions[symbol] = Position(
                symbol=symbol, quantity=quantity, entry_price=price, entry_date=datetime.now()
            )

        # Record trade
        self.trades_history.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.now(),
                "type": "buy" if quantity > 0 else "sell",
            }
        )

        return True

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        positions_value = sum(
            pos.quantity * current_prices[pos.symbol] for pos in self.positions.values()
        )
        return self.cash + positions_value

    def get_position_sizes(self) -> Dict[str, float]:
        """Get current position sizes"""
        return {symbol: position.quantity for symbol, position in self.positions.items()}

    def get_portfolio_metrics(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio metrics"""
        total_value = self.get_portfolio_value(current_prices)

        metrics = {
            "total_value": total_value,
            "cash": self.cash,
            "return": (total_value - self.initial_capital) / self.initial_capital,
        }

        # Calculate unrealized P&L for each position
        unrealized_pnl = 0
        for symbol, position in self.positions.items():
            current_price = current_prices[symbol]
            position_pnl = (current_price - position.entry_price) * position.quantity
            unrealized_pnl += position_pnl

        metrics["unrealized_pnl"] = unrealized_pnl

        return metrics
