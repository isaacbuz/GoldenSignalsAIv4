"""
Database Management for GoldenSignalsAI V3

Handles all database operations for agents, signals, and market data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    String, Integer, Float, DateTime, Boolean, JSON, Text,
    Index, ForeignKey, select, update, delete, and_, or_
)
from loguru import logger

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


class DatabaseManager:
    """
    Centralized database manager for all agent data operations
    """
    
    def __init__(self):
        self.engine = None
        self.async_session_factory = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connections and create tables"""
        database_url = settings.database.url
        
        # Auto-fallback logic: try PostgreSQL first, then SQLite
        if settings.database.db_type == "auto":
            try:
                logger.info("Attempting to connect to PostgreSQL...")
                await self._initialize_with_url(database_url)
                logger.info("✅ Connected to PostgreSQL database")
                return
            except Exception as pg_error:
                logger.warning(f"PostgreSQL connection failed: {str(pg_error)}")
                logger.info("Falling back to SQLite for development...")
                database_url = f"sqlite+aiosqlite:///{settings.database.sqlite_path}"
        
        # Direct initialization (either explicit type or fallback)
        try:
            await self._initialize_with_url(database_url)
            db_type = "SQLite" if "sqlite" in database_url else "PostgreSQL"
            logger.info(f"✅ Database manager initialized successfully with {db_type}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {str(e)}")
            raise
    
    async def _initialize_with_url(self, database_url: str) -> None:
        """Initialize database with a specific URL"""
        # Configure engine based on database type
        if "sqlite" in database_url:
            # SQLite: no connection pooling arguments allowed
            self.engine = create_async_engine(
                database_url,
                echo=settings.database.echo,
                poolclass=None
            )
        else:
            # PostgreSQL with connection pooling
            self.engine = create_async_engine(
                database_url,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                echo=settings.database.echo,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        
        # Create session factory
        self.async_session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create all tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        self._initialized = True
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup"""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.get_session() as session:
                result = await session.execute(select(1))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    # Signal operations
    async def store_signal(self, signal_data: Dict[str, Any]) -> str:
        """Store a signal in the database"""
        async with self.get_session() as session:
            signal_record = SignalRecord(**signal_data)
            session.add(signal_record)
            await session.flush()
            return signal_record.signal_id
    
    async def get_signals(
        self, 
        symbol: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[SignalRecord]:
        """Retrieve signals with optional filtering"""
        async with self.get_session() as session:
            query = select(SignalRecord)
            
            conditions = []
            if symbol:
                conditions.append(SignalRecord.symbol == symbol)
            if source:
                conditions.append(SignalRecord.source == source)
            if since:
                conditions.append(SignalRecord.created_at >= since)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            query = query.order_by(SignalRecord.created_at.desc()).limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def update_signal_performance(
        self, 
        signal_id: str, 
        performance_data: Dict[str, Any]
    ) -> bool:
        """Update signal performance metrics"""
        async with self.get_session() as session:
            query = (
                update(SignalRecord)
                .where(SignalRecord.signal_id == signal_id)
                .values(**performance_data)
            )
            result = await session.execute(query)
            return result.rowcount > 0
    
    # Meta-signal operations
    async def store_meta_signal(self, meta_signal_data: Dict[str, Any]) -> str:
        """Store a meta-signal in the database"""
        async with self.get_session() as session:
            meta_signal_record = MetaSignalRecord(**meta_signal_data)
            session.add(meta_signal_record)
            await session.flush()
            return meta_signal_record.meta_signal_id
    
    # Agent performance operations
    async def update_agent_performance(
        self, 
        agent_id: str, 
        performance_data: Dict[str, Any]
    ) -> None:
        """Update agent performance metrics"""
        async with self.get_session() as session:
            # Try to update existing record
            query = (
                update(AgentPerformanceRecord)
                .where(AgentPerformanceRecord.agent_id == agent_id)
                .values(**performance_data, last_updated=datetime.utcnow())
            )
            result = await session.execute(query)
            
            # If no record exists, create one
            if result.rowcount == 0:
                performance_record = AgentPerformanceRecord(
                    agent_id=agent_id,
                    **performance_data
                )
                session.add(performance_record)
    
    async def get_agent_performance(self, agent_id: str) -> Optional[AgentPerformanceRecord]:
        """Get agent performance metrics"""
        async with self.get_session() as session:
            query = select(AgentPerformanceRecord).where(
                AgentPerformanceRecord.agent_id == agent_id
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    # Agent state operations
    async def save_agent_state(
        self, 
        agent_id: str, 
        agent_name: str, 
        state_data: Dict[str, Any]
    ) -> None:
        """Save agent internal state"""
        async with self.get_session() as session:
            # Try to update existing state
            query = (
                update(AgentStateRecord)
                .where(AgentStateRecord.agent_id == agent_id)
                .values(
                    state_data=state_data,
                    last_updated=datetime.utcnow()
                )
            )
            result = await session.execute(query)
            
            # If no record exists, create one
            if result.rowcount == 0:
                state_record = AgentStateRecord(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    state_data=state_data
                )
                session.add(state_record)
    
    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent internal state"""
        async with self.get_session() as session:
            query = select(AgentStateRecord.state_data).where(
                AgentStateRecord.agent_id == agent_id
            )
            result = await session.execute(query)
            state_data = result.scalar_one_or_none()
            return state_data
    
    # Market data operations
    async def store_market_data(self, market_data_list: List[Dict[str, Any]]) -> None:
        """Store market data snapshots"""
        async with self.get_session() as session:
            records = [MarketDataRecord(**data) for data in market_data_list]
            session.add_all(records)
    
    async def get_market_data(
        self, 
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MarketDataRecord]:
        """Retrieve market data for a symbol"""
        async with self.get_session() as session:
            query = select(MarketDataRecord).where(MarketDataRecord.symbol == symbol)
            
            if since:
                query = query.where(MarketDataRecord.timestamp >= since)
            
            query = query.order_by(MarketDataRecord.timestamp.desc()).limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    # Analytics and reporting
    async def get_signal_analytics(
        self, 
        symbol: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get signal analytics for the past N days"""
        since = datetime.utcnow() - timedelta(days=days)
        
        async with self.get_session() as session:
            # Base query
            query = select(SignalRecord).where(SignalRecord.created_at >= since)
            if symbol:
                query = query.where(SignalRecord.symbol == symbol)
            
            result = await session.execute(query)
            signals = result.scalars().all()
            
            # Calculate analytics
            total_signals = len(signals)
            executed_signals = [s for s in signals if s.executed]
            profitable_signals = [s for s in executed_signals if s.was_profitable]
            
            analytics = {
                "total_signals": total_signals,
                "executed_signals": len(executed_signals),
                "profitable_signals": len(profitable_signals),
                "win_rate": len(profitable_signals) / len(executed_signals) if executed_signals else 0,
                "avg_confidence": sum(s.confidence for s in signals) / total_signals if signals else 0,
                "avg_return": sum(s.actual_return for s in executed_signals if s.actual_return) / len(executed_signals) if executed_signals else 0,
                "signals_by_source": {},
                "signals_by_type": {}
            }
            
            # Group by source and type
            for signal in signals:
                analytics["signals_by_source"][signal.source] = analytics["signals_by_source"].get(signal.source, 0) + 1
                analytics["signals_by_type"][signal.signal_type] = analytics["signals_by_type"].get(signal.signal_type, 0) + 1
            
            return analytics
    
    async def cleanup_old_data(self, days_to_keep: int = 90) -> None:
        """Clean up old data to prevent database bloat"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        async with self.get_session() as session:
            # Clean up old market data
            await session.execute(
                delete(MarketDataRecord).where(MarketDataRecord.timestamp < cutoff_date)
            )
            
            # Clean up old executed signals (keep unexecuted ones)
            await session.execute(
                delete(SignalRecord).where(
                    and_(
                        SignalRecord.created_at < cutoff_date,
                        SignalRecord.executed == True
                    )
                )
            )
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
    
    async def close(self) -> None:
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False
            logger.info("Database connections closed") 