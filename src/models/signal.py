"""
Signal model for storing trading signals
"""

import enum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import BaseModel


class SignalAction(enum.Enum):
    """Signal action types"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(enum.Enum):
    """Risk level types"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SignalStatus(enum.Enum):
    """Signal status types"""

    ACTIVE = "active"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class Signal(BaseModel):
    """Trading signal model"""

    __tablename__ = "signals"

    # Basic signal information
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(Enum(SignalAction), nullable=False, index=True)
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0

    # Price information
    price = Column(Float, nullable=False)
    target_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)

    # Risk and timing
    risk_level = Column(Enum(RiskLevel), nullable=False, index=True)
    timeframe = Column(String(20), default="1d", nullable=False)  # 1h, 4h, 1d, 1w
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # AI analysis
    indicators = Column(JSON, nullable=True)  # Technical indicators data
    reasoning = Column(Text, nullable=False)
    consensus_strength = Column(Float, nullable=False)  # Agent consensus strength
    agent_votes = Column(JSON, nullable=True)  # Individual agent votes

    # Signal metadata
    signal_source = Column(String(50), default="ai_consensus", nullable=False)
    market_regime = Column(String(20), nullable=True)  # bull, bear, sideways
    volatility_score = Column(Float, nullable=True)

    # Status and execution
    status = Column(Enum(SignalStatus), default=SignalStatus.ACTIVE, nullable=False, index=True)
    execution_price = Column(Float, nullable=True)
    executed_at = Column(DateTime(timezone=True), nullable=True)

    # Performance tracking
    pnl = Column(Float, default=0.0, nullable=False)  # Profit/Loss
    pnl_percentage = Column(Float, default=0.0, nullable=False)  # P&L percentage
    max_drawdown = Column(Float, nullable=True)
    max_profit = Column(Float, nullable=True)

    # View and interaction tracking
    views = Column(Integer, default=0, nullable=False)
    likes = Column(Integer, default=0, nullable=False)
    shares = Column(Integer, default=0, nullable=False)

    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    user = relationship("User", back_populates="signals")

    # Indexes for better query performance
    __table_args__ = (
        # Index for common queries
        {"extend_existing": True}
    )

    def __repr__(self):
        return f"<Signal(symbol={self.symbol}, action={self.action.value}, confidence={self.confidence:.2f})>"

    @property
    def is_active(self):
        """Check if signal is still active"""
        return self.status == SignalStatus.ACTIVE

    @property
    def is_profitable(self):
        """Check if signal is profitable"""
        return self.pnl > 0

    @property
    def signal_strength(self):
        """Calculate overall signal strength"""
        # Combine confidence and consensus strength
        return (self.confidence + self.consensus_strength) / 2

    @property
    def risk_reward_ratio(self):
        """Calculate risk-reward ratio"""
        if not self.stop_loss or not self.target_price:
            return None

        potential_profit = abs(self.target_price - self.price)
        potential_loss = abs(self.price - self.stop_loss)

        if potential_loss == 0:
            return None

        return potential_profit / potential_loss

    def update_performance(self, current_price: float):
        """Update signal performance metrics"""
        if self.execution_price:
            # Calculate P&L based on execution price
            if self.action == SignalAction.BUY:
                self.pnl = current_price - self.execution_price
            else:  # SELL
                self.pnl = self.execution_price - current_price

            self.pnl_percentage = (self.pnl / self.execution_price) * 100

            # Update max profit/drawdown
            if self.pnl > (self.max_profit or 0):
                self.max_profit = self.pnl
            if self.pnl < (self.max_drawdown or 0):
                self.max_drawdown = self.pnl

    def to_dict(self):
        """Convert to dictionary with computed fields"""
        data = super().to_dict()

        # Add computed properties
        data["is_active"] = self.is_active
        data["is_profitable"] = self.is_profitable
        data["signal_strength"] = self.signal_strength
        data["risk_reward_ratio"] = self.risk_reward_ratio

        # Convert enums to strings
        data["action"] = self.action.value if self.action else None
        data["risk_level"] = self.risk_level.value if self.risk_level else None
        data["status"] = self.status.value if self.status else None

        return data
