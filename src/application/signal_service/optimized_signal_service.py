"""
Optimized Signal Service with unified signal model and caching
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np
import redis.asyncio as redis
from agents.multi_agent_consensus import ConsensusResult, MultiAgentConsensus
from agents.multi_agent_consensus import SignalType as AgentSignalType
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.domain.trading.entities import Signal, SignalType
from src.infrastructure.caching import CacheManager
from src.websocket.signal_websocket import ws_service

logger = logging.getLogger(__name__)


class SignalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class UnifiedSignal:
    """Unified signal model for consistent data structure"""

    id: str
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float
    consensus_data: ConsensusResult
    contributing_agents: List[Dict[str, Any]]
    ai_reasoning: str
    metadata: Dict[str, Any]
    historical_accuracy: Optional[float] = None
    expected_move: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: SignalStatus = SignalStatus.ACTIVE
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
        data["consensus_data"] = {
            "consensus_id": self.consensus_data.consensus_id,
            "final_signal": self.consensus_data.final_signal.value,
            "confidence": self.consensus_data.confidence,
            "agreement_score": self.consensus_data.agreement_score,
            "participating_agents": self.consensus_data.participating_agents,
        }
        return data


class OptimizedSignalService:
    """
    Optimized signal service with caching and performance improvements
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        cache_manager: CacheManager,
        consensus_system: MultiAgentConsensus,
    ):
        self.redis = redis_client
        self.cache = cache_manager
        self.consensus = consensus_system

        # Performance settings
        self.batch_size = 50
        self.cache_ttl = 300  # 5 minutes
        self.hot_symbols_ttl = 60  # 1 minute

        # In-memory caches
        self.signal_cache: Dict[str, UnifiedSignal] = {}
        self.hot_symbols: Set[str] = set()
        self.accuracy_cache: Dict[str, float] = {}

        # Background tasks
        self._tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize the service and start background tasks"""
        # Start cache warming
        self._tasks.append(asyncio.create_task(self._warm_cache()))

        # Start accuracy updater
        self._tasks.append(asyncio.create_task(self._update_accuracy_cache()))

        # Start signal expiry checker
        self._tasks.append(asyncio.create_task(self._check_signal_expiry()))

        logger.info("Optimized Signal Service initialized")

    async def shutdown(self):
        """Cleanup resources"""
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def generate_signal(
        self, symbol: str, context: Dict[str, Any], priority: bool = False
    ) -> UnifiedSignal:
        """
        Generate a new signal with optimized performance
        """
        start_time = datetime.now()

        # Check if symbol is hot (frequently requested)
        if symbol in self.hot_symbols and not priority:
            # Use cached consensus if available
            cached = await self._get_cached_signal(symbol)
            if cached and (datetime.now() - cached.timestamp).seconds < 30:
                return cached

        # Request consensus from agents
        consensus = await self.consensus.request_consensus(
            symbol=symbol, context=context, timeout=2.0 if priority else 5.0
        )

        # Get historical accuracy for this type of signal
        accuracy = await self._get_historical_accuracy(symbol, consensus.final_signal.value)

        # Create unified signal
        signal = UnifiedSignal(
            id=f"SIG_{symbol}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=self._convert_signal_type(consensus.final_signal),
            confidence=consensus.confidence * 100,
            consensus_data=consensus,
            contributing_agents=[
                {
                    "agent_id": agent_id,
                    "type": self.consensus.registered_agents[agent_id]["type"].value,
                }
                for agent_id in consensus.participating_agents
            ],
            ai_reasoning=self._generate_reasoning(consensus, context),
            metadata={
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "context_summary": self._summarize_context(context),
                "market_conditions": context.get("regime", {}).get("type", "unknown"),
            },
            historical_accuracy=accuracy,
            expected_move=self._calculate_expected_move(consensus, context),
            stop_loss=consensus.risk_assessment.get("recommended_stop_loss"),
            take_profit=self._calculate_take_profit(consensus, context),
            expires_at=datetime.now() + timedelta(minutes=15),
        )

        # Cache the signal
        await self._cache_signal(signal)

        # Mark symbol as hot if generating multiple signals
        self.hot_symbols.add(symbol)

        # Broadcast via WebSocket
        await ws_service.broadcast_signal(signal.to_dict())

        # Update metrics
        await self._update_metrics(signal, start_time)

        return signal

    async def get_signal(self, signal_id: str) -> Optional[UnifiedSignal]:
        """Get a signal by ID with caching"""
        # Check in-memory cache
        if signal_id in self.signal_cache:
            return self.signal_cache[signal_id]

        # Check Redis cache
        cached = await self.redis.get(f"signal:{signal_id}")
        if cached:
            signal_data = json.loads(cached)
            return self._deserialize_signal(signal_data)

        # Load from database
        # TODO: Implement database query

        return None

    async def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        signal_type: Optional[SignalType] = None,
    ) -> List[UnifiedSignal]:
        """Get recent signals with filtering"""
        cache_key = f"recent_signals:{symbol or 'all'}:{signal_type or 'all'}:{limit}"

        # Check cache
        cached = await self.cache.get(cache_key)
        if cached:
            return [self._deserialize_signal(s) for s in cached]

        # Build query
        signals = []

        # Get from in-memory cache first
        for signal in list(self.signal_cache.values())[::-1]:  # Reverse for recent first
            if symbol and signal.symbol != symbol:
                continue
            if signal_type and signal.signal_type != signal_type:
                continue
            signals.append(signal)
            if len(signals) >= limit:
                break

        # Cache results
        await self.cache.set(cache_key, [s.to_dict() for s in signals], ttl=self.cache_ttl)

        return signals

    async def get_signal_analytics(self, timeframe: str = "24h") -> Dict[str, Any]:
        """Get signal analytics with caching"""
        cache_key = f"signal_analytics:{timeframe}"

        # Check cache
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        # Calculate analytics
        analytics = {
            "total_signals": len(self.signal_cache),
            "accuracy_by_type": {},
            "signal_distribution": {},
            "agent_performance": {},
            "avg_confidence": 0,
            "avg_processing_time": 0,
            "hot_symbols": list(self.hot_symbols)[:10],
        }

        # Calculate from cached signals
        if self.signal_cache:
            confidences = []
            processing_times = []
            signal_counts = defaultdict(int)

            for signal in self.signal_cache.values():
                confidences.append(signal.confidence)
                processing_times.append(signal.metadata.get("processing_time_ms", 0))
                signal_counts[signal.signal_type.value] += 1

            analytics["avg_confidence"] = np.mean(confidences)
            analytics["avg_processing_time"] = np.mean(processing_times)
            analytics["signal_distribution"] = dict(signal_counts)

        # Get accuracy data
        analytics["accuracy_by_type"] = dict(self.accuracy_cache)

        # Cache results
        await self.cache.set(cache_key, analytics, ttl=300)

        return analytics

    async def _get_cached_signal(self, symbol: str) -> Optional[UnifiedSignal]:
        """Get most recent cached signal for symbol"""
        for signal in reversed(list(self.signal_cache.values())):
            if signal.symbol == symbol and signal.status == SignalStatus.ACTIVE:
                return signal
        return None

    async def _cache_signal(self, signal: UnifiedSignal):
        """Cache signal in multiple layers"""
        # In-memory cache
        self.signal_cache[signal.id] = signal

        # Limit in-memory cache size
        if len(self.signal_cache) > 1000:
            # Remove oldest signals
            oldest_ids = sorted(
                self.signal_cache.keys(), key=lambda sid: self.signal_cache[sid].timestamp
            )[:100]
            for sid in oldest_ids:
                del self.signal_cache[sid]

        # Redis cache
        await self.redis.setex(f"signal:{signal.id}", self.cache_ttl, json.dumps(signal.to_dict()))

        # Add to recent signals list
        await self.redis.lpush(f"recent_signals:{signal.symbol}", signal.id)
        await self.redis.ltrim(f"recent_signals:{signal.symbol}", 0, 99)

    async def _get_historical_accuracy(self, symbol: str, signal_type: str) -> float:
        """Get historical accuracy for signal type"""
        cache_key = f"{symbol}:{signal_type}"

        if cache_key in self.accuracy_cache:
            return self.accuracy_cache[cache_key]

        # Default accuracy values (would come from database)
        default_accuracy = {"BUY": 0.925, "SELL": 0.918, "HOLD": 0.887}

        return default_accuracy.get(signal_type, 0.9)

    def _convert_signal_type(self, agent_signal: AgentSignalType) -> SignalType:
        """Convert agent signal type to unified signal type"""
        mapping = {
            AgentSignalType.STRONG_BUY: SignalType.BUY,
            AgentSignalType.BUY: SignalType.BUY,
            AgentSignalType.HOLD: SignalType.HOLD,
            AgentSignalType.SELL: SignalType.SELL,
            AgentSignalType.STRONG_SELL: SignalType.SELL,
            AgentSignalType.NO_SIGNAL: SignalType.HOLD,
        }
        return mapping.get(agent_signal, SignalType.HOLD)

    def _generate_reasoning(self, consensus: ConsensusResult, context: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for signal"""
        reasoning_parts = []

        # Add consensus strength
        if consensus.agreement_score > 0.8:
            reasoning_parts.append(f"Strong consensus ({consensus.agreement_score:.0%} agreement)")
        elif consensus.agreement_score > 0.6:
            reasoning_parts.append(
                f"Moderate consensus ({consensus.agreement_score:.0%} agreement)"
            )
        else:
            reasoning_parts.append(f"Weak consensus ({consensus.agreement_score:.0%} agreement)")

        # Add key factors
        if "sentiment" in context and context["sentiment"].get("score", 0) > 0.5:
            reasoning_parts.append("positive sentiment detected")
        elif "sentiment" in context and context["sentiment"].get("score", 0) < -0.5:
            reasoning_parts.append("negative sentiment detected")

        if "technical" in context:
            rsi = context["technical"].get("rsi", 50)
            if rsi < 30:
                reasoning_parts.append("oversold conditions")
            elif rsi > 70:
                reasoning_parts.append("overbought conditions")

        if "options_flow" in context:
            flow_score = context["options_flow"].get("smart_money_score", 50)
            if flow_score > 70:
                reasoning_parts.append("strong institutional buying")
            elif flow_score < 30:
                reasoning_parts.append("institutional selling pressure")

        return f"AI consensus based on {', '.join(reasoning_parts)}."

    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of context for metadata"""
        summary = {}

        for key in ["sentiment", "technical", "options_flow", "risk", "regime"]:
            if key in context:
                if isinstance(context[key], dict):
                    summary[key] = {
                        k: v
                        for k, v in context[key].items()
                        if k in ["score", "value", "type", "confidence"]
                    }
                else:
                    summary[key] = context[key]

        return summary

    def _calculate_expected_move(
        self, consensus: ConsensusResult, context: Dict[str, Any]
    ) -> float:
        """Calculate expected price move percentage"""
        base_move = 0.02  # 2% base move

        # Adjust based on confidence
        confidence_multiplier = consensus.confidence

        # Adjust based on market conditions
        volatility = context.get("risk", {}).get("volatility", 0.5)
        volatility_multiplier = 1 + volatility

        expected_move = base_move * confidence_multiplier * volatility_multiplier

        # Direction based on signal
        if consensus.final_signal in [AgentSignalType.SELL, AgentSignalType.STRONG_SELL]:
            expected_move *= -1

        return round(expected_move * 100, 2)  # Return as percentage

    def _calculate_take_profit(self, consensus: ConsensusResult, context: Dict[str, Any]) -> float:
        """Calculate take profit level"""
        expected_move = self._calculate_expected_move(consensus, context)

        # Take profit at 1.5x expected move
        return abs(expected_move) * 1.5

    def _deserialize_signal(self, data: Dict[str, Any]) -> UnifiedSignal:
        """Deserialize signal from dictionary"""
        # TODO: Implement proper deserialization
        # For now, return a mock signal
        return UnifiedSignal(
            id=data.get("id", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            symbol=data.get("symbol", ""),
            signal_type=SignalType(data.get("signal_type", "HOLD")),
            confidence=data.get("confidence", 0),
            consensus_data=data.get("consensus_data", {}),  # Would need proper deserialization
            contributing_agents=data.get("contributing_agents", []),
            ai_reasoning=data.get("ai_reasoning", ""),
            metadata=data.get("metadata", {}),
        )

    async def _warm_cache(self):
        """Background task to warm up caches"""
        while True:
            try:
                # Warm up accuracy cache
                # TODO: Load from database

                await asyncio.sleep(300)  # Every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(60)

    async def _update_accuracy_cache(self):
        """Background task to update accuracy metrics"""
        while True:
            try:
                # Calculate accuracy from recent signals
                # TODO: Implement accuracy calculation

                await asyncio.sleep(600)  # Every 10 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Accuracy update error: {e}")
                await asyncio.sleep(60)

    async def _check_signal_expiry(self):
        """Background task to expire old signals"""
        while True:
            try:
                now = datetime.now()
                expired = []

                for signal_id, signal in self.signal_cache.items():
                    if signal.expires_at and signal.expires_at < now:
                        signal.status = SignalStatus.EXPIRED
                        expired.append(signal_id)

                # Broadcast expiry notifications
                for signal_id in expired:
                    signal = self.signal_cache[signal_id]
                    await ws_service.broadcast_signal(
                        {**signal.to_dict(), "status": SignalStatus.EXPIRED.value}
                    )

                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Signal expiry check error: {e}")
                await asyncio.sleep(60)

    async def _update_metrics(self, signal: UnifiedSignal, start_time: datetime):
        """Update performance metrics"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Update Redis metrics
        await self.redis.hincrby("signal_metrics", "total_generated", 1)
        await self.redis.hincrbyfloat("signal_metrics", "total_processing_time", processing_time)
        await self.redis.hincrby(f"signal_metrics:{signal.signal_type.value}", "count", 1)

        # Publish metrics for monitoring
        await self.redis.publish(
            "metrics:signals",
            json.dumps(
                {
                    "signal_id": signal.id,
                    "symbol": signal.symbol,
                    "type": signal.signal_type.value,
                    "confidence": signal.confidence,
                    "processing_time_ms": processing_time,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
        )
