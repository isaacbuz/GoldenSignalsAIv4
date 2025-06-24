"""
Optimized API Endpoints for GoldenSignalsAI V2
Issue #197: Performance Optimization - Integrated API
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.infrastructure.caching.cache_manager import CacheManager, CacheStrategy
from src.infrastructure.database.query_optimizer import DatabaseOptimizer, QueryOptimizationStrategy
from src.infrastructure.performance.benchmark_suite import run_performance_benchmarks
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create optimized router
router = APIRouter(prefix="/api/v2", tags=["optimized"])

# Global instances (would be dependency injected in production)
cache_manager = CacheManager()
db_optimizer = DatabaseOptimizer("postgresql://localhost/goldensignals")


# Request/Response models
class SignalRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1, max_items=100)
    start_date: datetime
    end_date: datetime
    signal_types: Optional[List[str]] = None
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)


class SignalResponse(BaseModel):
    symbol: str
    timestamp: datetime
    signal_type: str
    confidence: float
    metadata: Dict[str, Any]


class PortfolioRequest(BaseModel):
    user_id: str
    include_history: bool = False
    history_days: int = Field(30, ge=1, le=365)


class MarketDataRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1, max_items=50)
    interval: str = Field("1m", regex="^(1m|5m|15m|30m|1h|1d)$")
    lookback_periods: int = Field(100, ge=1, le=1000)
    indicators: Optional[List[str]] = None


# Dependencies
async def get_cache() -> CacheManager:
    """Get cache manager instance"""
    return cache_manager


async def get_db() -> DatabaseOptimizer:
    """Get database optimizer instance"""
    return db_optimizer


# Optimized endpoints
@router.get("/signals", response_model=List[SignalResponse])
async def get_signals(
    request: SignalRequest,
    cache: CacheManager = Depends(get_cache),
    db: DatabaseOptimizer = Depends(get_db)
):
    """Get trading signals with optimized caching and query execution"""
    
    # Generate cache key
    cache_key = {
        "symbols": sorted(request.symbols),
        "start": request.start_date.isoformat(),
        "end": request.end_date.isoformat(),
        "types": sorted(request.signal_types) if request.signal_types else None,
        "min_conf": request.min_confidence
    }
    
    # Try cache first
    cached_result = await cache.get("signals", cache_key)
    if cached_result:
        logger.info(f"Cache hit for signals: {request.symbols}")
        return cached_result
    
    # Build optimized query
    query = """
        SELECT /*+ IndexScan(signals idx_signals_symbol_timestamp) */
            symbol, timestamp, signal_type, confidence, metadata
        FROM signals
        WHERE symbol = ANY($1)
        AND timestamp BETWEEN $2 AND $3
        AND confidence >= $4
    """
    
    params = [
        request.symbols,
        request.start_date,
        request.end_date,
        request.min_confidence
    ]
    
    if request.signal_types:
        query += " AND signal_type = ANY($5)"
        params.append(request.signal_types)
    
    query += " ORDER BY timestamp DESC"
    
    # Execute with optimization
    results = await db.execute_query(
        query,
        tuple(params),
        strategy=QueryOptimizationStrategy.INDEXED_LOOKUP
    )
    
    # Transform results
    signals = [
        SignalResponse(
            symbol=row["symbol"],
            timestamp=row["timestamp"],
            signal_type=row["signal_type"],
            confidence=row["confidence"],
            metadata=row.get("metadata", {})
        )
        for row in results
    ]
    
    # Cache results with appropriate TTL
    ttl = 300 if request.end_date.date() == datetime.now().date() else 3600
    await cache.set(
        "signals",
        cache_key,
        signals,
        ttl=ttl,
        strategy=CacheStrategy.REFRESH_AHEAD if ttl == 300 else CacheStrategy.TTL
    )
    
    return signals


@router.get("/market-data/stream")
async def stream_market_data(
    symbols: List[str] = Query(..., min_items=1, max_items=10),
    cache: CacheManager = Depends(get_cache),
    db: DatabaseOptimizer = Depends(get_db)
):
    """Stream real-time market data with optimized caching"""
    
    async def generate():
        """Generate market data stream"""
        while True:
            # Batch fetch for all symbols
            batch_key = f"realtime:{','.join(sorted(symbols))}"
            
            # Check cache for recent data
            cached_data = await cache.get("market_stream", batch_key, tier=cache.CacheTier.L1_MEMORY)
            
            if cached_data:
                yield f"data: {cached_data}\n\n"
            else:
                # Fetch from database with optimization
                query = """
                    SELECT symbol, price, volume, timestamp
                    FROM market_data
                    WHERE symbol = ANY($1)
                    AND timestamp > NOW() - INTERVAL '1 second'
                    ORDER BY timestamp DESC
                """
                
                results = await db.execute_query(
                    query,
                    (symbols,),
                    strategy=QueryOptimizationStrategy.INDEXED_LOOKUP,
                    use_cache=False  # Don't cache real-time data in query cache
                )
                
                if results:
                    # Cache in L1 only for 1 second
                    await cache.set(
                        "market_stream",
                        batch_key,
                        results,
                        ttl=1,
                        tier=cache.CacheTier.L1_MEMORY
                    )
                    
                    yield f"data: {results}\n\n"
            
            await asyncio.sleep(0.1)  # 100ms updates
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/portfolio/optimize")
async def optimize_portfolio(
    request: PortfolioRequest,
    cache: CacheManager = Depends(get_cache),
    db: DatabaseOptimizer = Depends(get_db)
):
    """Optimize portfolio with advanced caching and batch operations"""
    
    # Check if we have recent optimization
    cache_key = {
        "user_id": request.user_id,
        "include_history": request.include_history,
        "history_days": request.history_days
    }
    
    cached_result = await cache.get("portfolio_optimization", cache_key)
    if cached_result:
        return cached_result
    
    async with db.acquire_read() as conn:
        # Use prepared statement for performance
        if "get_portfolio_positions" not in db.prepared_statements:
            stmt = await conn.prepare("""
                SELECT symbol, quantity, avg_price, current_price
                FROM portfolio
                WHERE user_id = $1 AND active = true
            """)
            db.prepared_statements["get_portfolio_positions"] = stmt
        
        positions = await db.prepared_statements["get_portfolio_positions"].fetch(request.user_id)
    
    # Batch calculate optimizations
    optimization_tasks = []
    for position in positions:
        task = calculate_position_optimization(position, request.history_days)
        optimization_tasks.append(task)
    
    optimizations = await asyncio.gather(*optimization_tasks)
    
    result = {
        "user_id": request.user_id,
        "timestamp": datetime.now(),
        "positions": optimizations,
        "summary": calculate_portfolio_summary(optimizations)
    }
    
    # Cache with 5 minute TTL
    await cache.set(
        "portfolio_optimization",
        cache_key,
        result,
        ttl=300,
        strategy=CacheStrategy.WRITE_THROUGH
    )
    
    return result


@router.post("/batch/signals")
async def batch_create_signals(
    signals: List[SignalRequest],
    db: DatabaseOptimizer = Depends(get_db)
):
    """Batch create signals with optimized insertion"""
    
    # Prepare data for batch insert
    columns = ["symbol", "timestamp", "signal_type", "confidence", "metadata"]
    values = []
    
    for signal in signals:
        for symbol in signal.symbols:
            values.append((
                symbol,
                datetime.now(),
                signal.signal_types[0] if signal.signal_types else "AUTO",
                signal.min_confidence,
                {}
            ))
    
    # Batch insert with chunking
    inserted_count = await db.batch_insert(
        "signals",
        columns,
        values,
        chunk_size=1000
    )
    
    # Invalidate relevant caches
    for signal in signals:
        await cache_manager.delete_pattern("signals", f"*{signal.symbols}*")
    
    return {
        "status": "success",
        "inserted": inserted_count,
        "timestamp": datetime.now()
    }


@router.get("/analytics/performance")
async def get_performance_analytics(
    timeframe: str = Query("1d", regex="^(1h|1d|1w|1m)$"),
    cache: CacheManager = Depends(get_cache),
    db: DatabaseOptimizer = Depends(get_db)
):
    """Get performance analytics using materialized views"""
    
    # Map timeframe to view
    view_map = {
        "1h": "performance_hourly_mv",
        "1d": "performance_daily_mv",
        "1w": "performance_weekly_mv",
        "1m": "performance_monthly_mv"
    }
    
    view_name = view_map[timeframe]
    
    # Query materialized view (much faster than aggregating raw data)
    query = f"""
        SELECT * FROM {view_name}
        WHERE calculated_at > NOW() - INTERVAL '30 days'
        ORDER BY calculated_at DESC
    """
    
    results = await db.execute_query(
        query,
        strategy=QueryOptimizationStrategy.MATERIALIZED_VIEW,
        cache_ttl=3600  # Cache for 1 hour
    )
    
    return {
        "timeframe": timeframe,
        "data": results,
        "cached": len(results) > 0 and results[0].get("from_cache", False)
    }


@router.get("/health/performance")
async def health_check_performance(
    response: Response,
    cache: CacheManager = Depends(get_cache),
    db: DatabaseOptimizer = Depends(get_db)
):
    """Health check with performance metrics"""
    
    start_time = time.time()
    
    # Test cache performance
    cache_test_start = time.time()
    await cache.set("health", "test", {"status": "ok"}, ttl=1)
    cache_result = await cache.get("health", "test")
    cache_latency = (time.time() - cache_test_start) * 1000
    
    # Test database performance
    db_test_start = time.time()
    db_result = await db.execute_query("SELECT 1 as health", use_cache=False)
    db_latency = (time.time() - db_test_start) * 1000
    
    # Get metrics
    cache_metrics = await cache.get_metrics()
    db_metrics = await db.get_metrics()
    
    total_latency = (time.time() - start_time) * 1000
    
    # Set response headers
    response.headers["X-Response-Time"] = f"{total_latency:.2f}ms"
    response.headers["X-Cache-Hit-Rate"] = f"{cache_metrics.get('hit_rate', 0):.2f}%"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "latency": {
            "total_ms": total_latency,
            "cache_ms": cache_latency,
            "database_ms": db_latency
        },
        "cache": {
            "status": "ok" if cache_result else "error",
            "hit_rate": cache_metrics.get("hit_rate", 0),
            "l1_size": cache_metrics.get("l1_size", 0),
            "total_requests": cache_metrics.get("total_requests", 0)
        },
        "database": {
            "status": "ok" if db_result else "error",
            "connection_pool": db_metrics.get("read_pool", {}),
            "avg_query_time_ms": db_metrics.get("avg_query_time", 0) * 1000,
            "slow_queries": db_metrics.get("slow_queries", 0)
        }
    }


@router.post("/benchmark/run")
async def run_performance_benchmark(
    categories: Optional[List[str]] = None
):
    """Run performance benchmarks"""
    
    # Run benchmarks asynchronously
    task = asyncio.create_task(run_performance_benchmarks())
    
    return {
        "status": "started",
        "message": "Performance benchmarks running in background",
        "categories": categories or ["all"],
        "timestamp": datetime.now()
    }


# Helper functions
async def calculate_position_optimization(position: Dict[str, Any], history_days: int) -> Dict[str, Any]:
    """Calculate optimization for a single position"""
    # Simulate optimization calculation
    await asyncio.sleep(0.01)  # Simulate work
    
    return {
        "symbol": position["symbol"],
        "current_allocation": position["quantity"] * position["current_price"],
        "recommended_allocation": position["quantity"] * position["current_price"] * 1.1,
        "confidence": 0.85
    }


def calculate_portfolio_summary(optimizations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate portfolio summary statistics"""
    total_current = sum(opt["current_allocation"] for opt in optimizations)
    total_recommended = sum(opt["recommended_allocation"] for opt in optimizations)
    
    return {
        "total_value": total_current,
        "recommended_value": total_recommended,
        "expected_improvement": (total_recommended - total_current) / total_current * 100,
        "rebalance_required": abs(total_recommended - total_current) > total_current * 0.05
    }


# Startup event to initialize optimizations
@router.on_event("startup")
async def startup_optimizations():
    """Initialize performance optimizations on startup"""
    # Initialize cache
    await cache_manager.initialize()
    
    # Initialize database optimizer
    await db_optimizer.initialize()
    
    # Create materialized views for analytics
    await db_optimizer.create_materialized_view(
        "performance_daily_mv",
        """
        SELECT 
            DATE_TRUNC('day', created_at) as calculated_at,
            COUNT(*) as total_signals,
            AVG(confidence) as avg_confidence,
            SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
            SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals
        FROM signals
        GROUP BY DATE_TRUNC('day', created_at)
        """,
        indexes=["calculated_at"],
        refresh_interval=3600  # Refresh hourly
    )
    
    logger.info("Performance optimizations initialized")


# Shutdown event to cleanup
@router.on_event("shutdown")
async def shutdown_optimizations():
    """Cleanup on shutdown"""
    await cache_manager.close()
    await db_optimizer.close()
    logger.info("Performance optimizations cleaned up") 