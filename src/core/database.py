"""
Database Management for GoldenSignalsAI V3

Handles all database operations for agents, signals, and market data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Generator
from contextlib import asynccontextmanager
import os

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    String, Integer, Float, DateTime, Boolean, JSON, Text,
    Index, ForeignKey, select, update, delete, and_, or_, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger
from redis import Redis

from .config import settings


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class SignalRecord(Base):
    """Database model for storing signals"""
    
    __tablename__ = "signals"
    
    # Primary key and identification
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    
    # Basic signal information
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    signal_type: Mapped[str] = mapped_column(String(20), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    strength: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(50), index=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Price information
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Risk and position sizing
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    position_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Metadata
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    indicators: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON, nullable=True)
    market_conditions: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Performance tracking
    executed: Mapped[bool] = mapped_column(Boolean, default=False)
    execution_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    execution_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    actual_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    was_profitable: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_symbol_created', 'symbol', 'created_at'),
        Index('idx_source_created', 'source', 'created_at'),
        Index('idx_signal_type_created', 'signal_type', 'created_at'),
    )


class MetaSignalRecord(Base):
    """Database model for storing meta-signals (consensus signals)"""
    
    __tablename__ = "meta_signals"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meta_signal_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    
    # Consensus information
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    consensus_signal: Mapped[str] = mapped_column(String(20))
    consensus_confidence: Mapped[float] = mapped_column(Float)
    consensus_strength: Mapped[str] = mapped_column(String(20))
    
    # Component signals (JSON array of signal IDs)
    component_signal_ids: Mapped[List[str]] = mapped_column(JSON)
    agent_weights: Mapped[Dict[str, float]] = mapped_column(JSON)
    
    # Consensus metrics
    agreement_score: Mapped[float] = mapped_column(Float)
    uncertainty: Mapped[float] = mapped_column(Float)
    reliability: Mapped[float] = mapped_column(Float)
    
    # Aggregated financial metrics
    weighted_target_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weighted_stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recommended_position_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_meta_symbol_created', 'symbol', 'created_at'),
    )


class AgentPerformanceRecord(Base):
    """Database model for tracking agent performance"""
    
    __tablename__ = "agent_performance"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String(36), index=True)
    agent_name: Mapped[str] = mapped_column(String(100), index=True)
    
    # Performance metrics
    total_signals: Mapped[int] = mapped_column(Integer, default=0)
    correct_signals: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    avg_execution_time: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Weight and configuration
    current_weight: Mapped[float] = mapped_column(Float, default=1.0)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.7)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_agent_name_updated', 'agent_name', 'last_updated'),
    )


class MarketDataRecord(Base):
    """Database model for storing market data snapshots"""
    
    __tablename__ = "market_data"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    
    # OHLCV data
    open_price: Mapped[float] = mapped_column(Float)
    high_price: Mapped[float] = mapped_column(Float)
    low_price: Mapped[float] = mapped_column(Float)
    close_price: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)
    
    # Technical indicators (stored as JSON)
    indicators: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON, nullable=True)
    
    # Market conditions
    volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trend_direction: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    market_regime: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )


class AgentStateRecord(Base):
    """Database model for storing agent internal state"""
    
    __tablename__ = "agent_states"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String(36), index=True)
    agent_name: Mapped[str] = mapped_column(String(100), index=True)
    
    # State data
    state_data: Mapped[Dict[str, Any]] = mapped_column(JSON)
    
    # Model versions and checksums
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model_checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    
    # Timestamps
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_agent_state_updated', 'agent_id', 'last_updated'),
    )


# Change database URL to SQLite
DATABASE_URL = "sqlite:///./goldensignals.db"

# Create engine
engine = create_engine(DATABASE_URL)

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
Base.metadata.create_all(bind=engine)

class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    async def initialize(self):
        """Initialize database connection"""
        self.create_all()
        return self
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def create_all(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_all(self):
        """Drop all database tables"""
        Base.metadata.drop_all(bind=self.engine)
    
    async def close(self):
        """Close the database connection"""
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()

    async def health_check(self) -> bool:
        """Check the health of the database connection."""
        session = self.get_session()
        try:
            session.execute(select(1))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
        finally:
            session.close()

def get_redis() -> Redis:
    return Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True
    )

def get_db() -> Generator[Redis, None, None]:
    db = get_redis()
    try:
        yield db
    finally:
        db.close() 