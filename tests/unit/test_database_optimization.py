"""
Unit tests for Enhanced Database Query Optimizer
Issue #213: Database Query Optimization
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.infrastructure.database.enhanced_query_optimizer import (
    EnhancedDatabaseOptimizer,
    QueryPattern,
    IndexSuggestion,
    QueryStats
)
from src.services.database_optimization_service import (
    DatabaseOptimizationService
)


@pytest.fixture
async def mock_optimizer():
    """Create mock database optimizer"""
    optimizer = EnhancedDatabaseOptimizer(
        database_url="postgresql://test:test@localhost/test",
        redis_url=None,
        enable_query_cache=True,
        enable_auto_indexing=True
    )

    # Mock connection pools
    optimizer.read_pool = AsyncMock()
    optimizer.write_pool = AsyncMock()
    optimizer.redis_client = None

    return optimizer


@pytest.fixture
async def db_service(mock_optimizer):
    """Create database optimization service with mock"""
    with patch('src.services.database_optimization_service.EnhancedDatabaseOptimizer') as mock_cls:
        mock_cls.return_value = mock_optimizer

        await DatabaseOptimizationService.initialize(
            database_url="postgresql://test:test@localhost/test"
        )

        service = DatabaseOptimizationService.get_instance()
        service._optimizer = mock_optimizer

        yield service

        # Cleanup
        DatabaseOptimizationService._instance = None
        DatabaseOptimizationService._optimizer = None


class TestEnhancedDatabaseOptimizer:
    """Test enhanced database optimizer"""

    async def test_query_pattern_detection(self, mock_optimizer):
        """Test query pattern detection"""
        # Test point lookup
        pattern = mock_optimizer._detect_query_pattern(
            "SELECT * FROM signals WHERE id = $1"
        )
        assert pattern == QueryPattern.POINT_LOOKUP

        # Test range scan
        pattern = mock_optimizer._detect_query_pattern(
            "SELECT * FROM signals WHERE created_at BETWEEN $1 AND $2"
        )
        assert pattern == QueryPattern.RANGE_SCAN

        # Test aggregation
        pattern = mock_optimizer._detect_query_pattern(
            "SELECT symbol, COUNT(*) FROM signals GROUP BY symbol"
        )
        assert pattern == QueryPattern.AGGREGATION

        # Test join heavy
        pattern = mock_optimizer._detect_query_pattern(
            "SELECT s.*, o.* FROM signals s JOIN orders o ON s.id = o.signal_id"
        )
        assert pattern == QueryPattern.JOIN_HEAVY

    async def test_query_stats_tracking(self):
        """Test query statistics tracking"""
        stats = QueryStats()

        # Add executions
        stats.add_execution(0.5)
        stats.add_execution(1.5)
        stats.add_execution(0.3, cache_hit=True)

        assert stats.execution_count == 3
        assert stats.cache_hits == 1
        assert stats.slow_count == 1  # One query > 1.0s
        assert stats.min_time == 0.3
        assert stats.max_time == 1.5
        assert abs(stats.avg_time - 0.77) < 0.01

    async def test_index_suggestion(self):
        """Test index suggestion generation"""
        suggestion = IndexSuggestion(
            table="signals",
            columns=["symbol", "created_at"],
            query_pattern=QueryPattern.RANGE_SCAN
        )
        suggestion.benefit_score = 0.8

        sql = suggestion.to_sql()
        assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" in sql
        assert "idx_signals_symbol_created_at" in sql
        assert "ON signals (symbol, created_at)" in sql

    async def test_query_caching(self, mock_optimizer):
        """Test query result caching"""
        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Cache miss
        mock_redis.setex.return_value = True
        mock_optimizer.redis_client = mock_redis

        # Mock connection
        mock_conn = AsyncMock()
        mock_stmt = AsyncMock()
        mock_stmt.fetch.return_value = [{"id": 1, "symbol": "AAPL"}]
        mock_conn.prepare.return_value = mock_stmt

        mock_optimizer.read_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_optimizer.read_pool.release = AsyncMock()

        # Execute query
        result = await mock_optimizer.execute_optimized(
            query="SELECT * FROM signals WHERE symbol = $1",
            params=("AAPL",),
            use_cache=True
        )

        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"

        # Verify cache was set
        assert mock_redis.setex.called

        # Second call should hit cache
        mock_redis.get.return_value = json.dumps([{"id": 1, "symbol": "AAPL"}])

        result2 = await mock_optimizer.execute_optimized(
            query="SELECT * FROM signals WHERE symbol = $1",
            params=("AAPL",),
            use_cache=True
        )

        assert result2 == result
        # Should not prepare statement again
        assert mock_conn.prepare.call_count == 1

    async def test_batch_insert_optimization(self, mock_optimizer):
        """Test batch insert optimization"""
        records = [
            {"symbol": f"TEST{i}", "signal_type": "BUY", "confidence": 0.8}
            for i in range(100)
        ]

        mock_conn = AsyncMock()
        mock_conn.copy_records_to_table = AsyncMock()
        mock_optimizer.write_pool.acquire.return_value.__aenter__.return_value = mock_conn

        inserted = await mock_optimizer.batch_insert_optimized(
            table="signals",
            records=records,
            chunk_size=50
        )

        assert inserted == 100
        # Should be called twice (2 chunks of 50)
        assert mock_conn.copy_records_to_table.call_count == 2

    async def test_slow_query_detection(self, mock_optimizer):
        """Test slow query detection and logging"""
        # Mock slow query
        mock_conn = AsyncMock()
        mock_stmt = AsyncMock()

        # Simulate slow execution
        async def slow_fetch(*args):
            await asyncio.sleep(0.1)
            return []

        mock_stmt.fetch = slow_fetch
        mock_conn.prepare.return_value = mock_stmt
        mock_optimizer.read_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Set low threshold
        mock_optimizer.slow_query_threshold = 0.05

        # Execute query
        await mock_optimizer.execute_optimized(
            query="SELECT * FROM large_table",
            params=None
        )

        # Should be logged as slow
        assert len(mock_optimizer.slow_queries) == 1
        assert mock_optimizer.slow_queries[0]['execution_time'] > 0.05


class TestDatabaseOptimizationService:
    """Test database optimization service"""

    async def test_service_initialization(self):
        """Test service initialization"""
        with patch('src.services.database_optimization_service.EnhancedDatabaseOptimizer') as mock_cls:
            mock_optimizer = AsyncMock()
            mock_cls.return_value = mock_optimizer

            await DatabaseOptimizationService.initialize(
                database_url="postgresql://test:test@localhost/test",
                redis_url="redis://localhost:6379"
            )

            assert DatabaseOptimizationService._instance is not None
            assert DatabaseOptimizationService._optimizer is not None
            mock_optimizer.initialize.assert_called_once()

    async def test_query_pattern_auto_detection(self, db_service):
        """Test automatic query pattern detection"""
        # Test INSERT detection
        pattern = db_service._detect_query_pattern(
            "INSERT INTO signals (symbol, type) VALUES ($1, $2)"
        )
        assert pattern == QueryPattern.BATCH_OPERATION

        # Test UPDATE detection
        pattern = db_service._detect_query_pattern(
            "UPDATE signals SET confidence = $1 WHERE id = $2"
        )
        assert pattern == QueryPattern.BATCH_OPERATION

        # Test GROUP BY detection
        pattern = db_service._detect_query_pattern(
            "SELECT symbol, AVG(confidence) FROM signals GROUP BY symbol"
        )
        assert pattern == QueryPattern.AGGREGATION

    async def test_get_latest_signals(self, db_service):
        """Test get latest signals optimization"""
        # Mock optimizer response
        mock_signals = [
            {"id": 1, "symbol": "AAPL", "signal_type": "BUY"},
            {"id": 2, "symbol": "AAPL", "signal_type": "HOLD"}
        ]

        db_service._optimizer.execute_optimized = AsyncMock(return_value=mock_signals)

        signals = await db_service.get_latest_signals("AAPL", hours=24, limit=100)

        assert len(signals) == 2
        assert signals[0]["symbol"] == "AAPL"

        # Verify correct query pattern was used
        call_args = db_service._optimizer.execute_optimized.call_args
        assert call_args.kwargs['query_pattern'] == QueryPattern.RANGE_SCAN

    async def test_portfolio_summary(self, db_service):
        """Test portfolio summary optimization"""
        mock_summary = [{
            "position_count": 5,
            "total_value": 100000.0,
            "total_pnl": 5000.0,
            "winning_positions": 3
        }]

        db_service._optimizer.execute_optimized = AsyncMock(return_value=mock_summary)

        summary = await db_service.get_portfolio_summary(user_id=123)

        assert summary["position_count"] == 5
        assert summary["total_value"] == 100000.0
        assert summary["winning_positions"] == 3

        # Verify aggregation pattern was used
        call_args = db_service._optimizer.execute_optimized.call_args
        assert call_args.kwargs['query_pattern'] == QueryPattern.AGGREGATION

    async def test_batch_operations(self, db_service):
        """Test batch insert and update operations"""
        # Test batch insert
        signals = [
            {"symbol": "AAPL", "signal_type": "BUY", "confidence": 0.8},
            {"symbol": "GOOGL", "signal_type": "SELL", "confidence": 0.7}
        ]

        db_service._optimizer.batch_insert_optimized = AsyncMock(return_value=2)

        inserted = await db_service.batch_insert_signals(signals)
        assert inserted == 2

        # Test batch update
        updates = [
            {"id": 1, "current_price": 150.0},
            {"id": 2, "current_price": 2800.0}
        ]

        db_service._optimizer.bulk_update = AsyncMock(return_value=2)

        updated = await db_service.batch_update_portfolio(updates)
        assert updated == 2

        # Verify timestamp was added
        call_args = db_service._optimizer.bulk_update.call_args
        assert all('updated_at' in u for u in call_args.args[1])

    async def test_optimization_metrics(self, db_service):
        """Test getting optimization metrics"""
        mock_report = {
            "statistics": {
                "total_queries": 1000,
                "cache_hit_rate": 0.75,
                "slow_queries": 5
            },
            "index_suggestions": []
        }

        db_service._optimizer.get_optimization_report = AsyncMock(return_value=mock_report)
        db_service._optimizer.query_stats = {"q1": QueryStats(), "q2": QueryStats()}
        db_service._optimizer.prepared_statements = {"q1": "stmt1"}
        db_service._optimizer.index_suggestions = [
            IndexSuggestion("signals", ["symbol"], QueryPattern.POINT_LOOKUP)
        ]

        metrics = await db_service.get_optimization_metrics()

        assert metrics["statistics"]["total_queries"] == 1000
        assert metrics["statistics"]["cache_hit_rate"] == 0.75
        assert "application_metrics" in metrics
        assert metrics["application_metrics"]["cached_queries"] == 2
        assert metrics["application_metrics"]["prepared_statements"] == 1

    async def test_apply_suggested_indexes(self, db_service):
        """Test applying suggested indexes"""
        # Create suggestions
        suggestion1 = IndexSuggestion("signals", ["symbol"], QueryPattern.POINT_LOOKUP)
        suggestion1.benefit_score = 0.9

        suggestion2 = IndexSuggestion("orders", ["user_id"], QueryPattern.POINT_LOOKUP)
        suggestion2.benefit_score = 0.6  # Below threshold

        db_service._optimizer.index_suggestions = [suggestion1, suggestion2]
        db_service._optimizer.create_smart_index = AsyncMock(return_value=True)

        applied = await db_service.apply_suggested_indexes(min_benefit_score=0.7)

        assert len(applied) == 1
        assert applied[0]["table"] == "signals"
        assert applied[0]["benefit_score"] == 0.9

        # Only high-benefit index should be created
        db_service._optimizer.create_smart_index.assert_called_once()

    async def test_table_analysis(self, db_service):
        """Test analyzing all tables"""
        mock_analysis = {
            "table": "signals",
            "statistics": {"row_count": 1000000},
            "indexes": [],
            "recommendations": ["Consider VACUUM"]
        }

        db_service._optimizer.analyze_table_statistics = AsyncMock(return_value=mock_analysis)

        analyses = await db_service.analyze_all_tables()

        assert "signals" in analyses
        assert analyses["signals"]["statistics"]["row_count"] == 1000000
        assert len(analyses["signals"]["recommendations"]) == 1

        # Should analyze all important tables
        assert db_service._optimizer.analyze_table_statistics.call_count == 4


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for database optimization"""

    async def test_end_to_end_optimization(self, db_service):
        """Test end-to-end optimization flow"""
        # Mock data
        db_service._optimizer.execute_optimized = AsyncMock()
        db_service._optimizer.query_stats["test_query"] = QueryStats()
        db_service._optimizer.query_stats["test_query"].add_execution(2.0)  # Slow query

        # Add index suggestion
        suggestion = IndexSuggestion("signals", ["symbol", "created_at"], QueryPattern.RANGE_SCAN)
        suggestion.benefit_score = 0.85
        db_service._optimizer.index_suggestions = [suggestion]

        # Mock methods
        db_service._optimizer.get_optimization_report = AsyncMock(return_value={
            "statistics": {"slow_queries": 15},
            "index_suggestions": []
        })
        db_service._optimizer.create_smart_index = AsyncMock(return_value=True)

        # Simulate periodic task behavior
        report = await db_service.get_optimization_metrics()

        if report["statistics"]["slow_queries"] > 10:
            applied = await db_service.apply_suggested_indexes(min_benefit_score=0.8)
            assert len(applied) == 1

    async def test_transaction_context(self, db_service):
        """Test transaction context manager"""
        mock_conn = AsyncMock()
        db_service._optimizer.write_pool.acquire.return_value.__aenter__.return_value = mock_conn

        async with db_service.transaction() as conn:
            assert conn == mock_conn

        # Verify transaction was used
        mock_conn.transaction.assert_called()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
