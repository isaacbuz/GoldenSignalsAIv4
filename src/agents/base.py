"""
Base Agent Architecture for GoldenSignalsAI V3
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.database import DatabaseManager
from src.core.redis_manager import RedisManager
from src.models.signals import Signal, SignalStrength, SignalType
from src.models.market_data import MarketData
from src.utils.metrics import MetricsCollector


class AgentConfig(BaseModel):
    """Configuration for an individual agent"""
    
    name: str
    version: str = "1.0.0"
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    timeout: int = Field(default=30, gt=0)  # seconds
    max_retries: int = Field(default=3, ge=0)
    learning_rate: float = Field(default=0.01, gt=0.0)


class AgentPerformance(BaseModel):
    """Performance metrics for an agent"""
    
    agent_id: str
    total_signals: int = 0
    correct_signals: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    avg_execution_time: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def update_accuracy(self) -> float:
        """Update and return current accuracy"""
        if self.total_signals > 0:
            self.accuracy = self.correct_signals / self.total_signals
        return self.accuracy


class BaseAgent(ABC):
    """
    Base class for all trading agents in GoldenSignalsAI V3
    
    Features:
    - Asynchronous execution
    - Performance tracking and metrics
    - Adaptive confidence scoring
    - Error handling and retries
    - Self-learning capabilities
    """
    
    def __init__(self, config: AgentConfig, db_manager: DatabaseManager, redis_manager: RedisManager):
        self.config = config
        self.agent_id = str(uuid.uuid4())
        self.performance = AgentPerformance(agent_id=self.agent_id)
        self.metrics = MetricsCollector(f"agent_{config.name}")
        self._is_running = False
        self._stop_event = asyncio.Event()
        
        # Data layer managers
        self.db_manager = db_manager
        self.redis_manager = redis_manager
        
        # Agent state for persistence
        self._state = {
            "model_parameters": {},
            "learned_patterns": {},
            "performance_history": [],
            "last_analysis_time": None,
            "cached_calculations": {}
        }
        
        logger.info(f"Initialized {config.name} agent with ID: {self.agent_id}")
    
    @abstractmethod
    async def analyze(self, *args, **kwargs) -> Signal:
        """
        Core analysis method that each agent must implement
        
        Args:
            *args / **kwargs: Implementation-specific market data inputs
            
        Returns:
            Signal: Trading signal with confidence and metadata
        """
        pass
    
    @abstractmethod
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for this agent
        
        Returns:
            List of data type strings (e.g., ['price', 'volume', 'news'])
        """
        pass
    
    async def execute_with_monitoring(self, market_data: MarketData) -> Optional[Signal]:
        """
        Execute analysis with comprehensive monitoring and error handling
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            Signal if successful, None if failed
        """
        if not self.config.enabled:
            logger.debug(f"Agent {self.config.name} is disabled")
            return None
        
        start_time = time.time()
        attempt = 0
        
        while attempt < self.config.max_retries:
            try:
                # Execute with timeout
                signal = await asyncio.wait_for(
                    self.analyze(market_data),
                    timeout=self.config.timeout
                )
                
                # Record successful execution
                execution_time = time.time() - start_time
                await self._record_success(signal, execution_time)
                
                logger.debug(
                    f"Agent {self.config.name} generated signal: "
                    f"{signal.signal_type.value} with confidence {signal.confidence:.3f}"
                )
                
                return signal
                
            except asyncio.TimeoutError:
                logger.warning(f"Agent {self.config.name} timed out on attempt {attempt + 1}")
                attempt += 1
                
            except Exception as e:
                logger.error(f"Agent {self.config.name} failed: {str(e)}")
                attempt += 1
                
                if attempt >= self.config.max_retries:
                    await self._record_failure(e)
                    break
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    async def _record_success(self, signal: Signal, execution_time: float) -> None:
        """Record successful signal generation"""
        self.performance.total_signals += 1
        self.performance.avg_execution_time = (
            (self.performance.avg_execution_time * (self.performance.total_signals - 1) + execution_time)
            / self.performance.total_signals
        )
        self.performance.avg_confidence = (
            (self.performance.avg_confidence * (self.performance.total_signals - 1) + signal.confidence)
            / self.performance.total_signals
        )
        self.performance.last_updated = datetime.utcnow()
        
        # Store signal in database
        signal_data = {
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "signal_type": signal.signal_type.value,
            "confidence": signal.confidence,
            "strength": signal.strength.value,
            "source": self.config.name,
            "current_price": signal.current_price,
            "target_price": signal.target_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_score": signal.risk_score,
            "reasoning": signal.reasoning,
            "features": signal.features,
            "indicators": signal.indicators,
            "market_conditions": signal.market_conditions
        }
        await self.db_manager.store_signal(signal_data)
        
        # Cache signal for real-time access
        await self.redis_manager.add_signal_to_stream(signal.symbol, signal_data)
        
        # Update agent performance in database
        performance_data = {
            "agent_name": self.config.name,
            "total_signals": self.performance.total_signals,
            "correct_signals": self.performance.correct_signals,
            "accuracy": self.performance.accuracy,
            "avg_confidence": self.performance.avg_confidence,
            "avg_execution_time": self.performance.avg_execution_time,
            "current_weight": self.config.weight,
            "confidence_threshold": self.config.confidence_threshold,
            "is_enabled": self.config.enabled
        }
        await self.db_manager.update_agent_performance(self.agent_id, performance_data)
        
        # Cache performance for quick access
        await self.redis_manager.cache_agent_performance(self.agent_id, performance_data)
        
        # Update metrics
        self.metrics.increment("signals_generated")
        self.metrics.record("execution_time", execution_time)
        self.metrics.record("confidence", signal.confidence)
        
    async def _record_failure(self, error: Exception) -> None:
        """Record failed signal generation"""
        self.metrics.increment("signals_failed")
        logger.error(f"Agent {self.config.name} failed: {str(error)}")
    
    def update_performance_feedback(self, signal_id: str, was_correct: bool) -> None:
        """
        Update agent performance based on signal outcome
        
        Args:
            signal_id: ID of the signal to update
            was_correct: Whether the signal prediction was correct
        """
        if was_correct:
            self.performance.correct_signals += 1
        
        # Update accuracy
        self.performance.update_accuracy()
        
        # Adaptive learning: adjust confidence threshold based on performance
        if self.performance.accuracy < 0.5 and self.performance.total_signals > 10:
            # If performing poorly, increase confidence threshold
            self.config.confidence_threshold = min(0.9, self.config.confidence_threshold + self.config.learning_rate)
        elif self.performance.accuracy > 0.75:
            # If performing well, decrease confidence threshold slightly
            self.config.confidence_threshold = max(0.5, self.config.confidence_threshold - self.config.learning_rate / 2)
        
        logger.debug(
            f"Agent {self.config.name} performance updated: "
            f"accuracy={self.performance.accuracy:.3f}, "
            f"confidence_threshold={self.config.confidence_threshold:.3f}"
        )
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "accuracy": self.performance.accuracy,
            "total_signals": self.performance.total_signals,
            "avg_confidence": self.performance.avg_confidence,
            "avg_execution_time": self.performance.avg_execution_time,
            "weight": self.config.weight,
            "confidence_threshold": self.config.confidence_threshold,
            "enabled": self.config.enabled
        }
    
    def adjust_weight(self, new_weight: float) -> None:
        """Adjust agent weight based on performance"""
        self.config.weight = max(0.0, min(1.0, new_weight))
        logger.info(f"Agent {self.config.name} weight adjusted to {self.config.weight:.3f}")
    
    async def start(self) -> None:
        """Start the agent"""
        self._is_running = True
        self._stop_event.clear()
        logger.info(f"Agent {self.config.name} started")
    
    async def stop(self) -> None:
        """Stop the agent"""
        self._is_running = False
        self._stop_event.set()
        logger.info(f"Agent {self.config.name} stopped")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent"""
        await self.stop()
        # Save current state before shutdown
        await self.save_state()
        logger.info(f"Agent {self.config.name} shutdown complete")
    
    @property
    def is_running(self) -> bool:
        """Check if agent is running"""
        return self._is_running
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, weight={self.config.weight})"
    
    async def save_state(self) -> None:
        """Save agent internal state to database"""
        self._state["last_analysis_time"] = datetime.utcnow().isoformat()
        await self.db_manager.save_agent_state(self.agent_id, self.config.name, self._state)
        
        # Also cache temporarily for quick access
        await self.redis_manager.cache_agent_state(self.agent_id, self._state)
    
    async def load_state(self) -> None:
        """Load agent internal state from database"""
        # Try cache first, then database
        state_data = await self.redis_manager.get_cached_agent_state(self.agent_id)
        
        if not state_data:
            state_data = await self.db_manager.load_agent_state(self.agent_id)
        
        if state_data:
            self._state.update(state_data)
            logger.info(f"Loaded state for agent {self.config.name}")
    
    async def get_historical_market_data(
        self, 
        symbol: str, 
        timeframe: str = "1h", 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical market data for analysis"""
        # Try cache first
        cached_data = await self.redis_manager.get_cached_ohlcv_data(symbol, timeframe)
        
        if cached_data:
            return cached_data[:limit]
        
        # Fallback to database
        since = datetime.utcnow() - timedelta(hours=limit)
        market_records = await self.db_manager.get_market_data(symbol, since, limit)
        
        # Convert to dict format and cache
        data = []
        for record in market_records:
            data.append({
                "timestamp": record.timestamp.isoformat(),
                "open": record.open_price,
                "high": record.high_price,
                "low": record.low_price,
                "close": record.close_price,
                "volume": record.volume,
                "indicators": record.indicators or {}
            })
        
        # Cache for future use
        if data:
            await self.redis_manager.cache_ohlcv_data(symbol, timeframe, data)
        
        return data
    
    async def get_recent_signals(
        self, 
        symbol: str, 
        source: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent signals for analysis"""
        # Try cache first
        if not source:
            cached_signals = await self.redis_manager.get_cached_latest_signals(symbol)
            if cached_signals:
                return cached_signals[:limit]
        
        # Get from database
        since = datetime.utcnow() - timedelta(hours=24)  # Last 24 hours
        signal_records = await self.db_manager.get_signals(
            symbol=symbol, 
            source=source or self.config.name, 
            limit=limit, 
            since=since
        )
        
        # Convert to dict format
        signals = []
        for record in signal_records:
            signals.append({
                "signal_id": record.signal_id,
                "symbol": record.symbol,
                "signal_type": record.signal_type,
                "confidence": record.confidence,
                "strength": record.strength,
                "source": record.source,
                "created_at": record.created_at.isoformat(),
                "reasoning": record.reasoning,
                "features": record.features,
                "indicators": record.indicators
            })
        
        return signals
    
    async def update_learning_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """Update agent learning parameters"""
        self._state["model_parameters"].update(new_parameters)
        await self.save_state()
        logger.info(f"Updated learning parameters for agent {self.config.name}")
    
    async def cache_calculation(self, key: str, value: Any, ttl: int = 300) -> None:
        """Cache expensive calculations"""
        cache_key = f"{self.agent_id}:{key}"
        await self.redis_manager.store_temp_data(cache_key, value, ttl)
    
    async def get_cached_calculation(self, key: str) -> Any:
        """Get cached calculation"""
        cache_key = f"{self.agent_id}:{key}"
        return await self.redis_manager.get_temp_data(cache_key)
    
    async def get_peer_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance data from other agents for ensemble learning"""
        # This would be implemented by the orchestrator
        # For now, return empty dict
        return {}

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional for subclasses)
    # ------------------------------------------------------------------

    async def initialize(self) -> None:  # noqa: D401
        """Optional async initialization hook.

        Sub-classes can override if they need to load ML models, warm-up caches,
        etc.  The base implementation is a no-op so the orchestrator can safely
        `await agent.initialize()` for every agent type without having to check
        whether the method exists.
        """
        return 