"""
Portfolio Models - GoldenSignalsAI V3

SQLAlchemy models for portfolio management and tracking.
"""

import enum
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import relationship

from .base import BaseModel


class TradeType(enum.Enum):
    """Trade type enumeration"""
    BUY = "buy"
    SELL = "sell"


class TradeStatus(enum.Enum):
    """Trade status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionType(enum.Enum):
    """Position type enumeration"""
    LONG = "long"
    SHORT = "short"


class Portfolio(BaseModel):
    """
    Portfolio model for tracking overall portfolio performance
    """
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Portfolio values
    initial_capital = Column(Numeric(15, 2), nullable=False)
    current_value = Column(Numeric(15, 2), nullable=False)
    cash_balance = Column(Numeric(15, 2), nullable=False)
    buying_power = Column(Numeric(15, 2), nullable=False)
    
    # Performance metrics
    total_return = Column(Numeric(10, 4), default=0.0)
    total_return_percent = Column(Numeric(8, 4), default=0.0)
    day_change = Column(Numeric(15, 2), default=0.0)
    day_change_percent = Column(Numeric(8, 4), default=0.0)
    
    # Risk metrics
    sharpe_ratio = Column(Numeric(8, 4), nullable=True)
    max_drawdown = Column(Numeric(8, 4), nullable=True)
    volatility = Column(Numeric(8, 4), nullable=True)
    beta = Column(Numeric(8, 4), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_paper_trading = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="portfolio", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Portfolio(id={self.id}, name='{self.name}', value={self.current_value})>"
    
    @property
    def total_positions_value(self) -> Decimal:
        """Calculate total value of all positions"""
        return sum(pos.market_value for pos in self.positions if pos.is_active)
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions if pos.is_active)
    
    @property
    def positions_count(self) -> int:
        """Get count of active positions"""
        return len([pos for pos in self.positions if pos.is_active])


class Position(BaseModel):
    """
    Position model for tracking individual stock positions
    """
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Position details
    position_type = Column(Enum(PositionType), nullable=False)
    quantity = Column(Numeric(15, 6), nullable=False)
    avg_cost = Column(Numeric(15, 4), nullable=False)
    current_price = Column(Numeric(15, 4), nullable=False)
    
    # Calculated values
    market_value = Column(Numeric(15, 2), nullable=False)
    unrealized_pnl = Column(Numeric(15, 2), nullable=False)
    unrealized_pnl_percent = Column(Numeric(8, 4), nullable=False)
    day_change = Column(Numeric(15, 2), default=0.0)
    day_change_percent = Column(Numeric(8, 4), default=0.0)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    trades = relationship("Trade", back_populates="position")
    
    def __repr__(self):
        return f"<Position(id={self.id}, symbol='{self.symbol}', quantity={self.quantity}, pnl={self.unrealized_pnl})>"
    
    def update_market_value(self, current_price: Decimal):
        """Update position with current market price"""
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        
        # Calculate P&L
        cost_basis = self.quantity * self.avg_cost
        self.unrealized_pnl = self.market_value - cost_basis
        
        if cost_basis != 0:
            self.unrealized_pnl_percent = (self.unrealized_pnl / cost_basis) * 100
        else:
            self.unrealized_pnl_percent = 0
        
        self.updated_at = datetime.utcnow()


class Trade(BaseModel):
    """
    Trade model for tracking individual trade executions
    """
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=True)
    
    # Trade details
    symbol = Column(String(20), nullable=False, index=True)
    trade_type = Column(Enum(TradeType), nullable=False)
    quantity = Column(Numeric(15, 6), nullable=False)
    price = Column(Numeric(15, 4), nullable=False)
    total_value = Column(Numeric(15, 2), nullable=False)
    
    # Fees and costs
    commission = Column(Numeric(10, 2), default=0.0)
    fees = Column(Numeric(10, 2), default=0.0)
    
    # Order details
    order_id = Column(String(100), nullable=True)
    execution_id = Column(String(100), nullable=True)
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional info
    notes = Column(Text, nullable=True)
    strategy = Column(String(100), nullable=True)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="trades")
    position = relationship("Position", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol='{self.symbol}', type='{self.trade_type.value}', quantity={self.quantity}, price={self.price})>"
    
    @property
    def net_value(self) -> Decimal:
        """Calculate net trade value after fees"""
        return self.total_value - self.commission - self.fees
    
    def mark_executed(self, execution_price: Optional[Decimal] = None):
        """Mark trade as executed"""
        self.status = TradeStatus.FILLED
        self.executed_at = datetime.utcnow()
        
        if execution_price:
            self.price = execution_price
            self.total_value = self.quantity * execution_price
        
        self.updated_at = datetime.utcnow() 