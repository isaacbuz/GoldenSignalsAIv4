"""
Portfolio model for tracking user portfolios and positions
"""

import enum

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import BaseModel


class PortfolioType(enum.Enum):
    """Portfolio types"""

    PAPER = "paper"  # Paper trading
    LIVE = "live"  # Live trading
    BACKTEST = "backtest"  # Backtesting


class Portfolio(BaseModel):
    """User portfolio model"""

    __tablename__ = "portfolios"

    # Basic portfolio information
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    portfolio_type = Column(String(20), default="paper", nullable=False)

    # Portfolio value
    initial_balance = Column(Float, default=10000.0, nullable=False)
    current_balance = Column(Float, default=10000.0, nullable=False)
    total_value = Column(Float, default=10000.0, nullable=False)  # cash + positions

    # Performance metrics
    total_pnl = Column(Float, default=0.0, nullable=False)
    total_pnl_percentage = Column(Float, default=0.0, nullable=False)
    daily_pnl = Column(Float, default=0.0, nullable=False)

    # Risk metrics
    max_drawdown = Column(Float, default=0.0, nullable=False)
    max_drawdown_percentage = Column(Float, default=0.0, nullable=False)
    volatility = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)

    # Trading statistics
    total_trades = Column(Integer, default=0, nullable=False)
    winning_trades = Column(Integer, default=0, nullable=False)
    losing_trades = Column(Integer, default=0, nullable=False)
    win_rate = Column(Float, default=0.0, nullable=False)

    # Position management
    max_positions = Column(Integer, default=10, nullable=False)
    current_positions = Column(Integer, default=0, nullable=False)
    position_sizing = Column(String(20), default="equal", nullable=False)  # equal, kelly, fixed

    # Risk management
    max_risk_per_trade = Column(Float, default=0.02, nullable=False)  # 2% max risk
    max_portfolio_risk = Column(Float, default=0.1, nullable=False)  # 10% max portfolio risk
    stop_loss_percentage = Column(Float, default=0.05, nullable=False)  # 5% stop loss

    # Portfolio settings
    auto_execute = Column(Boolean, default=False, nullable=False)
    follow_signals = Column(Boolean, default=True, nullable=False)
    risk_management_enabled = Column(Boolean, default=True, nullable=False)

    # Holdings and history
    positions = Column(JSON, nullable=True)  # Current positions
    trade_history = Column(JSON, nullable=True)  # Historical trades
    performance_history = Column(JSON, nullable=True)  # Daily performance

    # Activity tracking
    is_active = Column(Boolean, default=True, nullable=False)
    last_trade_at = Column(DateTime(timezone=True), nullable=True)
    last_updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", back_populates="portfolios")

    def __repr__(self):
        return f"<Portfolio(name={self.name}, value=${self.total_value:.2f}, pnl={self.total_pnl_percentage:.2f}%)>"

    @property
    def profit_factor(self):
        """Calculate profit factor"""
        if self.losing_trades > 0:
            avg_win = self.total_pnl / max(self.winning_trades, 1) if self.winning_trades > 0 else 0
            avg_loss = abs(self.total_pnl) / self.losing_trades if self.total_pnl < 0 else 1
            return avg_win / avg_loss if avg_loss > 0 else 0
        return float("inf") if self.winning_trades > 0 else 0

    @property
    def available_cash(self):
        """Calculate available cash for new positions"""
        return self.current_balance

    @property
    def position_value(self):
        """Calculate total value of current positions"""
        return self.total_value - self.current_balance

    @property
    def is_profitable(self):
        """Check if portfolio is profitable"""
        return self.total_pnl > 0

    @property
    def risk_score(self):
        """Calculate portfolio risk score (0-100)"""
        risk_factors = []

        # Drawdown factor
        if self.max_drawdown_percentage > 0:
            risk_factors.append(min(self.max_drawdown_percentage * 2, 50))

        # Volatility factor
        if self.volatility:
            risk_factors.append(min(self.volatility * 100, 30))

        # Position concentration
        if self.current_positions > 0:
            concentration = self.current_positions / self.max_positions
            risk_factors.append(concentration * 20)

        return sum(risk_factors) if risk_factors else 0

    def add_trade(
        self, symbol: str, action: str, quantity: int, price: float, signal_id: str = None
    ):
        """Add a new trade to the portfolio"""
        trade = {
            "id": str(self.id),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": quantity * price,
            "timestamp": func.now().isoformat(),
            "signal_id": signal_id,
        }

        # Update trade history
        if not self.trade_history:
            self.trade_history = []
        self.trade_history.append(trade)

        # Update portfolio metrics
        self.total_trades += 1
        self.last_trade_at = func.now()

        # Update positions
        self.update_positions(symbol, action, quantity, price)

    def update_positions(self, symbol: str, action: str, quantity: int, price: float):
        """Update portfolio positions"""
        if not self.positions:
            self.positions = {}

        if action.upper() == "BUY":
            if symbol in self.positions:
                # Average price calculation
                current_qty = self.positions[symbol]["quantity"]
                current_price = self.positions[symbol]["avg_price"]
                total_cost = (current_qty * current_price) + (quantity * price)
                total_qty = current_qty + quantity

                self.positions[symbol] = {
                    "quantity": total_qty,
                    "avg_price": total_cost / total_qty,
                    "current_price": price,
                    "last_updated": func.now().isoformat(),
                }
            else:
                self.positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": price,
                    "current_price": price,
                    "last_updated": func.now().isoformat(),
                }
                self.current_positions += 1

            # Update cash balance
            self.current_balance -= quantity * price

        elif action.upper() == "SELL":
            if symbol in self.positions:
                current_qty = self.positions[symbol]["quantity"]

                if quantity >= current_qty:
                    # Close position
                    del self.positions[symbol]
                    self.current_positions -= 1
                else:
                    # Partial close
                    self.positions[symbol]["quantity"] -= quantity

                # Update cash balance
                self.current_balance += quantity * price

                # Calculate P&L for this trade
                avg_price = self.positions.get(symbol, {}).get("avg_price", price)
                trade_pnl = (price - avg_price) * quantity
                self.total_pnl += trade_pnl

                if trade_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

    def update_performance(self, current_prices: dict):
        """Update portfolio performance with current market prices"""
        total_position_value = 0

        # Update position values
        if self.positions:
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    position["current_price"] = current_price
                    position_value = position["quantity"] * current_price
                    total_position_value += position_value

        # Update total portfolio value
        self.total_value = self.current_balance + total_position_value

        # Calculate P&L
        self.total_pnl = self.total_value - self.initial_balance
        self.total_pnl_percentage = (self.total_pnl / self.initial_balance) * 100

        # Update win rate
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100

        # Update performance history
        performance_entry = {
            "date": func.now().date().isoformat(),
            "total_value": self.total_value,
            "pnl": self.total_pnl,
            "pnl_percentage": self.total_pnl_percentage,
            "positions": dict(self.positions) if self.positions else {},
        }

        if not self.performance_history:
            self.performance_history = []
        self.performance_history.append(performance_entry)

        # Keep only last 365 days
        if len(self.performance_history) > 365:
            self.performance_history = self.performance_history[-365:]

    def calculate_position_size(self, symbol: str, price: float, risk_amount: float):
        """Calculate position size based on risk management"""
        if self.position_sizing == "equal":
            # Equal weight across max positions
            position_value = self.total_value / self.max_positions
            return int(position_value / price)

        elif self.position_sizing == "fixed":
            # Fixed dollar amount
            return int(risk_amount / price)

        elif self.position_sizing == "kelly":
            # Kelly criterion (simplified)
            win_rate = self.win_rate / 100 if self.win_rate > 0 else 0.5
            avg_win = self.total_pnl / max(self.winning_trades, 1) if self.winning_trades > 0 else 1
            avg_loss = (
                abs(self.total_pnl) / max(self.losing_trades, 1) if self.losing_trades > 0 else 1
            )

            if avg_loss > 0:
                kelly_percentage = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                kelly_percentage = max(0, min(kelly_percentage, 0.25))  # Cap at 25%
                position_value = self.total_value * kelly_percentage
                return int(position_value / price)

        return 0

    def to_dict(self):
        """Convert to dictionary with computed fields"""
        data = super().to_dict()

        # Add computed properties
        data["profit_factor"] = self.profit_factor
        data["available_cash"] = self.available_cash
        data["position_value"] = self.position_value
        data["is_profitable"] = self.is_profitable
        data["risk_score"] = self.risk_score

        return data
