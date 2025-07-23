"""
Database Optimization Service for GoldenSignalsAI V4
Issue #213: Database Query Optimization
Integrates the enhanced optimizer with the application
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI

from src.infrastructure.database.enhanced_query_optimizer import (
    EnhancedDatabaseOptimizer,
    IndexSuggestion,
    QueryPattern,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseOptimizationService:
    """Service to manage database optimization across the application"""

    _instance: Optional["DatabaseOptimizationService"] = None
    _optimizer: Optional[EnhancedDatabaseOptimizer] = None

    @classmethod
    async def initialize(cls, database_url: str, redis_url: Optional[str] = None):
        """Initialize the singleton service"""
        if cls._instance is None:
            cls._instance = cls()
            cls._optimizer = EnhancedDatabaseOptimizer(
                database_url=database_url,
                redis_url=redis_url,
                enable_query_cache=True,
                enable_auto_indexing=True,
                slow_query_threshold=1.0,
                cache_ttl=300,
            )
            await cls._optimizer.initialize()
            logger.info("Database optimization service initialized")

    @classmethod
    def get_instance(cls) -> "DatabaseOptimizationService":
        """Get the singleton instance"""
        if cls._instance is None:
            raise RuntimeError("DatabaseOptimizationService not initialized")
        return cls._instance

    @classmethod
    def get_optimizer(cls) -> EnhancedDatabaseOptimizer:
        """Get the optimizer instance"""
        if cls._optimizer is None:
            raise RuntimeError("Optimizer not initialized")
        return cls._optimizer

    async def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        use_cache: bool = True,
        query_pattern: Optional[QueryPattern] = None,
    ) -> List[Dict[str, Any]]:
        """Execute an optimized query"""
        # Auto-detect query pattern if not provided
        if query_pattern is None:
            query_pattern = self._detect_query_pattern(query)

        return await self._optimizer.execute_optimized(
            query=query, params=params, query_pattern=query_pattern, use_cache=use_cache
        )

    def _detect_query_pattern(self, query: str) -> QueryPattern:
        """Auto-detect query pattern from SQL"""
        query_upper = query.upper()

        if "INSERT" in query_upper or "UPDATE" in query_upper:
            return QueryPattern.BATCH_OPERATION
        elif "GROUP BY" in query_upper:
            return QueryPattern.AGGREGATION
        elif "JOIN" in query_upper:
            return QueryPattern.JOIN_HEAVY
        elif "BETWEEN" in query_upper or ">" in query_upper or "<" in query_upper:
            return QueryPattern.RANGE_SCAN
        else:
            return QueryPattern.POINT_LOOKUP

    async def get_latest_signals(
        self, symbol: str, hours: int = 24, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get latest signals for a symbol (optimized)"""
        return await self.execute_query(
            query="""
                SELECT * FROM signals
                WHERE symbol = $1 AND created_at > $2
                ORDER BY created_at DESC
                LIMIT $3
            """,
            params=(symbol, datetime.now() - timedelta(hours=hours), limit),
            query_pattern=QueryPattern.RANGE_SCAN,
        )

    async def get_portfolio_summary(self, user_id: int) -> Dict[str, Any]:
        """Get portfolio summary (optimized)"""
        result = await self.execute_query(
            query="""
                SELECT
                    COUNT(DISTINCT symbol) as position_count,
                    SUM(quantity * current_price) as total_value,
                    SUM(quantity * current_price - quantity * avg_cost) as total_pnl,
                    SUM(CASE WHEN quantity * current_price > quantity * avg_cost THEN 1 ELSE 0 END) as winning_positions
                FROM portfolio
                WHERE user_id = $1 AND active = true
            """,
            params=(user_id,),
            query_pattern=QueryPattern.AGGREGATION,
        )

        return result[0] if result else {}

    async def get_signal_performance(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get signal performance metrics (optimized)"""
        return await self.execute_query(
            query="""
                SELECT
                    s.signal_type,
                    COUNT(*) as signal_count,
                    AVG(s.confidence) as avg_confidence,
                    COUNT(o.id) as executed_count,
                    AVG(o.profit_loss) as avg_profit
                FROM signals s
                LEFT JOIN orders o ON s.id = o.signal_id
                WHERE s.created_at BETWEEN $1 AND $2
                GROUP BY s.signal_type
                ORDER BY signal_count DESC
            """,
            params=(start_date, end_date),
            query_pattern=QueryPattern.JOIN_HEAVY,
        )

    async def batch_insert_signals(self, signals: List[Dict[str, Any]]) -> int:
        """Batch insert signals (optimized)"""
        return await self._optimizer.batch_insert_optimized(
            table="signals",
            records=signals,
            on_conflict="ON CONFLICT (symbol, signal_type, created_at) DO UPDATE SET confidence = EXCLUDED.confidence",
        )

    async def batch_update_portfolio(self, updates: List[Dict[str, Any]]) -> int:
        """Batch update portfolio positions"""
        # Prepare updates with current timestamp
        for update in updates:
            update["updated_at"] = datetime.now()

        return await self._optimizer.bulk_update(
            table="portfolio", updates=updates, key_column="id"
        )

    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get database optimization metrics"""
        report = await self._optimizer.get_optimization_report()

        # Add application-specific metrics
        report["application_metrics"] = {
            "cached_queries": len(self._optimizer.query_stats),
            "prepared_statements": len(self._optimizer.prepared_statements),
            "pending_index_suggestions": len(
                [s for s in self._optimizer.index_suggestions if s.benefit_score > 0.5]
            ),
        }

        return report

    async def apply_suggested_indexes(self, min_benefit_score: float = 0.7):
        """Apply high-benefit index suggestions"""
        applied = []

        for suggestion in self._optimizer.index_suggestions:
            if suggestion.benefit_score >= min_benefit_score:
                success = await self._optimizer.create_smart_index(suggestion)
                if success:
                    applied.append(
                        {
                            "table": suggestion.table,
                            "columns": suggestion.columns,
                            "benefit_score": suggestion.benefit_score,
                        }
                    )

        return applied

    async def analyze_all_tables(self) -> Dict[str, Any]:
        """Analyze all important tables"""
        tables = ["signals", "portfolio", "orders", "market_data"]
        analyses = {}

        for table in tables:
            try:
                analyses[table] = await self._optimizer.analyze_table_statistics(table)
            except Exception as e:
                logger.error(f"Failed to analyze table {table}: {e}")
                analyses[table] = {"error": str(e)}

        return analyses

    @asynccontextmanager
    async def transaction(self):
        """Get a database transaction context"""
        async with self._optimizer.write_pool.acquire() as conn:
            async with conn.transaction():
                yield conn


# FastAPI integration
def setup_database_optimization(app: FastAPI):
    """Setup database optimization for FastAPI app"""

    @app.on_event("startup")
    async def startup_db_optimization():
        """Initialize database optimization on startup"""
        database_url = app.state.config.database.url
        redis_url = app.state.config.redis.url

        await DatabaseOptimizationService.initialize(database_url=database_url, redis_url=redis_url)

        # Schedule periodic optimization tasks
        asyncio.create_task(periodic_optimization_tasks())

    @app.on_event("shutdown")
    async def shutdown_db_optimization():
        """Clean up on shutdown"""
        optimizer = DatabaseOptimizationService.get_optimizer()
        await optimizer.close()


async def periodic_optimization_tasks():
    """Run periodic optimization tasks"""
    service = DatabaseOptimizationService.get_instance()

    while True:
        try:
            # Every hour: Apply beneficial indexes
            await asyncio.sleep(3600)

            # Get optimization report
            report = await service.get_optimization_metrics()
            logger.info(f"DB Optimization Report: {report['statistics']}")

            # Apply high-benefit indexes automatically
            if report["statistics"]["slow_queries"] > 10:
                applied = await service.apply_suggested_indexes(min_benefit_score=0.8)
                if applied:
                    logger.info(f"Applied {len(applied)} new indexes")

            # Analyze tables if needed
            if report["statistics"]["slow_queries"] > 50:
                analyses = await service.analyze_all_tables()
                for table, analysis in analyses.items():
                    if "recommendations" in analysis:
                        for rec in analysis["recommendations"]:
                            logger.warning(f"Table {table}: {rec}")

        except Exception as e:
            logger.error(f"Error in periodic optimization: {e}")
            await asyncio.sleep(60)  # Retry after a minute


from datetime import datetime

# Example usage in API endpoints
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/v1", tags=["optimized"])


@router.get("/signals/{symbol}")
async def get_signals_optimized(
    symbol: str, hours: int = Query(24, ge=1, le=168), limit: int = Query(100, ge=1, le=1000)
):
    """Get signals with database optimization"""
    try:
        service = DatabaseOptimizationService.get_instance()
        signals = await service.get_latest_signals(symbol, hours, limit)

        return {"symbol": symbol, "count": len(signals), "signals": signals}
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/{user_id}/summary")
async def get_portfolio_summary_optimized(user_id: int):
    """Get portfolio summary with optimization"""
    try:
        service = DatabaseOptimizationService.get_instance()
        summary = await service.get_portfolio_summary(user_id)

        return {"user_id": user_id, "summary": summary, "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/db-optimization/metrics")
async def get_db_optimization_metrics():
    """Get database optimization metrics (admin only)"""
    try:
        service = DatabaseOptimizationService.get_instance()
        metrics = await service.get_optimization_metrics()

        return {"status": "healthy", "metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/db-optimization/apply-indexes")
async def apply_indexes(min_benefit_score: float = Query(0.7, ge=0.0, le=1.0)):
    """Apply suggested indexes (admin only)"""
    try:
        service = DatabaseOptimizationService.get_instance()
        applied = await service.apply_suggested_indexes(min_benefit_score)

        return {"applied_count": len(applied), "indexes": applied}
    except Exception as e:
        logger.error(f"Error applying indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
