"""
Signal Service - GoldenSignalsAI V3

Business logic for trading signal operations, including storage, retrieval,
analytics, and performance tracking.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

from src.core.database import DatabaseManager
from src.core.redis_manager import RedisManager
from src.ml.models.signals import Signal, SignalType, SignalStrength


class SignalService:
    """
    Service for managing trading signals and their lifecycle
    """
    
    def __init__(self, db_manager: DatabaseManager, redis_manager: RedisManager):
        self.db_manager = db_manager
        self.redis_manager = redis_manager
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the signal service"""
        try:
            # Verify database and Redis connections
            if not await self.db_manager.health_check():
                raise RuntimeError("Database connection failed")
            
            if not await self.redis_manager.health_check():
                raise RuntimeError("Redis connection failed")
            
            self._initialized = True
            logger.info("SignalService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SignalService: {str(e)}")
            raise
    
    async def store_signal(self, signal: Signal) -> str:
        """
        Store a new trading signal
        
        Args:
            signal: Signal object to store
            
        Returns:
            str: Signal ID
        """
        if not self._initialized:
            raise RuntimeError("SignalService not initialized")
        
        try:
            # Convert signal to database format
            signal_data = {
                "signal_id": signal.signal_id,
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "confidence": signal.confidence,
                "strength": signal.strength.value,
                "source": signal.source,
                "current_price": signal.current_price,
                "target_price": signal.target_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "risk_score": signal.risk_score,
                "reasoning": signal.reasoning,
                "features": signal.features,
                "indicators": signal.indicators,
                "market_conditions": signal.market_conditions,
                "expires_at": signal.expires_at
            }
            
            # Store in database
            signal_id = await self.db_manager.store_signal(signal_data)
            
            # Cache for real-time access
            await self.redis_manager.add_signal_to_stream(signal.symbol, signal_data)
            
            # Publish to subscribers
            await self.redis_manager.publish_signal(signal.symbol, signal_data)
            
            logger.debug(f"Stored signal {signal_id} for {signal.symbol}")
            return signal_id
            
        except Exception as e:
            logger.error(f"Failed to store signal: {str(e)}")
            raise
    
    async def get_signals_paginated(
        self,
        symbol: Optional[str] = None,
        source: Optional[str] = None,
        signal_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        since: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Any], int]:
        """
        Get paginated signals with filtering
        
        Args:
            symbol: Filter by symbol
            source: Filter by source agent
            signal_type: Filter by signal type
            min_confidence: Minimum confidence threshold
            since: Filter signals created after this time
            page: Page number
            page_size: Items per page
            
        Returns:
            Tuple of (signals list, total count)
        """
        try:
            # Get signals from database
            all_signals = await self.db_manager.get_signals(
                symbol=symbol,
                source=source,
                limit=page_size * 10,  # Get more for filtering
                since=since
            )
            
            # Apply additional filters
            filtered_signals = []
            for signal in all_signals:
                # Filter by signal type
                if signal_type and signal.signal_type != signal_type:
                    continue
                
                # Filter by confidence
                if min_confidence and signal.confidence < min_confidence:
                    continue
                
                filtered_signals.append(signal)
            
            # Apply pagination
            total = len(filtered_signals)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_signals = filtered_signals[start_idx:end_idx]
            
            return paginated_signals, total
            
        except Exception as e:
            logger.error(f"Failed to get paginated signals: {str(e)}")
            raise
    
    async def get_latest_signals(
        self,
        symbol: str,
        limit: int = 10
    ) -> List[Any]:
        """
        Get the latest signals for a symbol
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of signals
            
        Returns:
            List of latest signals
        """
        try:
            # Try Redis cache first
            cached_signals = await self.redis_manager.get_cached_latest_signals(symbol)
            if cached_signals:
                return cached_signals[:limit]
            
            # Fallback to database
            signals = await self.db_manager.get_signals(
                symbol=symbol,
                limit=limit
            )
            
            # Cache the results
            if signals:
                signal_data = [
                    {
                        "signal_id": s.signal_id,
                        "symbol": s.symbol,
                        "signal_type": s.signal_type,
                        "confidence": s.confidence,
                        "strength": s.strength,
                        "source": s.source,
                        "created_at": s.created_at.isoformat(),
                        "reasoning": s.reasoning
                    }
                    for s in signals
                ]
                await self.redis_manager.cache_latest_signals(symbol, signal_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to get latest signals: {str(e)}")
            raise
    
    async def get_signal_stream(
        self,
        symbol: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get real-time signal stream for a symbol
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of signals
            
        Returns:
            List of signals in stream format
        """
        try:
            return await self.redis_manager.get_signal_stream(symbol, limit)
            
        except Exception as e:
            logger.error(f"Failed to get signal stream: {str(e)}")
            raise
    
    async def get_signal_analytics(
        self,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get signal analytics and performance metrics
        
        Args:
            symbol: Optional symbol filter
            days: Number of days to analyze
            
        Returns:
            Analytics dictionary
        """
        try:
            return await self.db_manager.get_signal_analytics(symbol, days)
            
        except Exception as e:
            logger.error(f"Failed to get signal analytics: {str(e)}")
            raise
    
    async def update_signal_performance(
        self,
        signal_id: str,
        was_profitable: bool,
        actual_return: Optional[float] = None,
        notes: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update signal performance with trading results
        
        Args:
            signal_id: ID of the signal
            was_profitable: Whether the trade was profitable
            actual_return: Actual return percentage
            notes: Additional notes
            user_id: User providing feedback
            
        Returns:
            bool: Success status
        """
        try:
            performance_data = {
                "executed": True,
                "was_profitable": was_profitable,
                "actual_return": actual_return,
                "execution_time": datetime.utcnow(),
                "feedback_notes": notes,
                "feedback_user": user_id
            }
            
            success = await self.db_manager.update_signal_performance(
                signal_id,
                performance_data
            )
            
            if success:
                logger.info(f"Updated performance for signal {signal_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update signal performance: {str(e)}")
            raise
    
    async def get_signal_by_id(self, signal_id: str) -> Optional[Any]:
        """
        Get a specific signal by ID
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            Signal record or None
        """
        try:
            signals = await self.db_manager.get_signals(limit=1)
            for signal in signals:
                if signal.signal_id == signal_id:
                    return signal
            return None
            
        except Exception as e:
            logger.error(f"Failed to get signal by ID: {str(e)}")
            raise
    
    async def cleanup_expired_signals(self) -> int:
        """
        Clean up expired signals from cache
        
        Returns:
            int: Number of signals cleaned up
        """
        try:
            # This would be implemented to clean up expired signals
            # For now, return 0
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired signals: {str(e)}")
            raise
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get overall signal service performance metrics
        
        Returns:
            Performance metrics dictionary
        """
        try:
            # Get cache statistics
            cache_stats = await self.redis_manager.get_cache_stats()
            
            # Get recent signal counts
            recent_signals = await self.db_manager.get_signals(limit=1000)
            
            return {
                "total_signals_24h": len([
                    s for s in recent_signals 
                    if s.created_at > datetime.utcnow() - timedelta(hours=24)
                ]),
                "cache_hit_rate": cache_stats.get("cache_hit_rate", 0),
                "avg_signal_confidence": sum(s.confidence for s in recent_signals) / len(recent_signals) if recent_signals else 0,
                "service_status": "healthy" if self._initialized else "unhealthy"
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the signal service"""
        try:
            self._initialized = False
            logger.info("SignalService shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during SignalService shutdown: {str(e)}") 