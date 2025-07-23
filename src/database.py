"""
Database configuration and session management
"""

import logging
import os

from sqlalchemy import MetaData, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from models.base import Base

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/goldensignals")
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create engines
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False  # Set to True for SQL logging
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False  # Set to True for SQL logging
)

# Session makers
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Metadata
metadata = MetaData()


def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        raise


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db():
    """Dependency to get async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


class DatabaseManager:
    """Database manager for handling connections and operations"""
    
    def __init__(self):
        self.engine = engine
        self.async_engine = async_engine
        self.session_local = SessionLocal
        self.async_session_local = AsyncSessionLocal
    
    def create_tables(self):
        """Create all database tables"""
        create_tables()
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.session_local()
    
    async def get_async_session(self) -> AsyncSession:
        """Get a new async database session"""
        return self.async_session_local()
    
    def health_check(self) -> bool:
        """Check database connection health"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def async_health_check(self) -> bool:
        """Check async database connection health"""
        try:
            async with self.async_session_local() as session:
                await session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Async database health check failed: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        try:
            self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    async def async_close(self):
        """Close async database connections"""
        try:
            await self.async_engine.dispose()
            logger.info("Async database connections closed")
        except Exception as e:
            logger.error(f"Error closing async database connections: {e}")


# Global database manager instance
db_manager = DatabaseManager()


# Database utilities
def init_database():
    """Initialize database with tables and initial data"""
    try:
        logger.info("🔄 Initializing database...")
        
        # Create tables
        create_tables()
        
        # Create initial agents if they don't exist
        from models.agent import Agent
        
        with SessionLocal() as session:
            existing_agents = session.query(Agent).count()
            
            if existing_agents == 0:
                logger.info("Creating initial AI agents...")
                
                initial_agents = [
                    {
                        "name": "RSI_Agent",
                        "agent_type": "rsi",
                        "description": "Relative Strength Index based trading agent",
                        "config": {"period": 14, "oversold": 30, "overbought": 70}
                    },
                    {
                        "name": "MACD_Agent", 
                        "agent_type": "macd",
                        "description": "MACD indicator based trading agent",
                        "config": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                    },
                    {
                        "name": "Sentiment_Agent",
                        "agent_type": "sentiment", 
                        "description": "Market sentiment analysis agent",
                        "config": {"sources": ["news", "social", "options_flow"]}
                    },
                    {
                        "name": "Volume_Agent",
                        "agent_type": "volume",
                        "description": "Volume analysis trading agent", 
                        "config": {"volume_threshold": 1.5, "volume_period": 10}
                    },
                    {
                        "name": "Momentum_Agent",
                        "agent_type": "momentum",
                        "description": "Price momentum analysis agent",
                        "config": {"lookback_period": 5, "momentum_threshold": 2.0}
                    }
                ]
                
                for agent_data in initial_agents:
                    agent = Agent(**agent_data)
                    session.add(agent)
                
                session.commit()
                logger.info(f"✅ Created {len(initial_agents)} initial agents")
        
        logger.info("✅ Database initialization completed")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


async def async_init_database():
    """Async version of database initialization"""
    try:
        logger.info("🔄 Initializing database (async)...")
        
        # Create tables
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ Async database initialization completed")
        
    except Exception as e:
        logger.error(f"❌ Async database initialization failed: {e}")
        raise


# Export commonly used items
__all__ = [
    "engine",
    "async_engine", 
    "SessionLocal",
    "AsyncSessionLocal",
    "get_db",
    "get_async_db",
    "DatabaseManager",
    "db_manager",
    "create_tables",
    "init_database",
    "async_init_database"
] 