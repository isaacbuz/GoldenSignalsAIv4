"""
Database Query Optimizer for GoldenSignalsAI V2
Issue #197: Performance Optimization - Database Query Optimization
"""

import asyncio
import hashlib
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import asyncpg
from asyncpg.pool import Pool
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryOptimizationStrategy(Enum):
    """Query optimization strategies"""
    BATCH_INSERT = "batch_insert"
    BULK_UPDATE = "bulk_update"
    INDEXED_LOOKUP = "indexed_lookup"
    MATERIALIZED_VIEW = "materialized_view"
    PARTITION_SCAN = "partition_scan"
    PREPARED_STATEMENT = "prepared_statement"


class ConnectionPoolConfig:
    """Connection pool configuration"""
    def __init__(
        self,
        min_size: int = 10,
        max_size: int = 50,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0,
        command_timeout: float = 60.0,
        max_cached_statement_lifetime: int = 300,
        max_cacheable_statement_size: int = 1024 * 15
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime
        self.command_timeout = command_timeout
        self.max_cached_statement_lifetime = max_cached_statement_lifetime
        self.max_cacheable_statement_size = max_cacheable_statement_size


class DatabaseOptimizer:
    """Advanced database optimizer with connection pooling and query optimization"""
    
    def __init__(
        self,
        database_url: str,
        pool_config: Optional[ConnectionPoolConfig] = None,
        enable_query_cache: bool = True,
        enable_metrics: bool = True
    ):
        self.database_url = database_url
        self.pool_config = pool_config or ConnectionPoolConfig()
        self.enable_query_cache = enable_query_cache
        self.enable_metrics = enable_metrics
        
        # Connection pools
        self.read_pool: Optional[Pool] = None
        self.write_pool: Optional[Pool] = None
        self.async_engine: Optional[AsyncEngine] = None
        
        # Query cache
        self.query_cache: Dict[str, Tuple[Any, float]] = {}
        self.prepared_statements: Dict[str, str] = {}
        
        # Metrics
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "slow_queries": 0,
            "connection_waits": 0,
            "pool_exhausted": 0,
            "query_errors": 0,
            "avg_query_time": 0.0
        }
        
        # Query timing
        self.query_times: List[float] = []
        
    async def initialize(self):
        """Initialize database connection pools"""
        try:
            # Create read pool (larger for read-heavy workloads)
            self.read_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.pool_config.min_size,
                max_size=self.pool_config.max_size,
                max_queries=self.pool_config.max_queries,
                max_inactive_connection_lifetime=self.pool_config.max_inactive_connection_lifetime,
                command_timeout=self.pool_config.command_timeout,
                max_cached_statement_lifetime=self.pool_config.max_cached_statement_lifetime,
                max_cacheable_statement_size=self.pool_config.max_cacheable_statement_size
            )
            
            # Create write pool (smaller to prevent connection exhaustion)
            self.write_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=max(2, self.pool_config.min_size // 2),
                max_size=max(10, self.pool_config.max_size // 3),
                max_queries=self.pool_config.max_queries,
                command_timeout=self.pool_config.command_timeout
            )
            
            # Create SQLAlchemy async engine for ORM operations
            self.async_engine = create_async_engine(
                self.database_url.replace("postgresql://", "postgresql+asyncpg://"),
                pool_size=self.pool_config.min_size,
                max_overflow=self.pool_config.max_size - self.pool_config.min_size,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Initialize database optimizations
            await self._initialize_optimizations()
            
            logger.info("Database optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database optimizer: {e}")
            raise
    
    async def close(self):
        """Close database connections"""
        if self.read_pool:
            await self.read_pool.close()
        if self.write_pool:
            await self.write_pool.close()
        if self.async_engine:
            await self.async_engine.dispose()
    
    async def _initialize_optimizations(self):
        """Initialize database optimizations"""
        async with self.write_pool.acquire() as conn:
            # Enable query planner optimizations
            await conn.execute("SET enable_seqscan = OFF")
            await conn.execute("SET random_page_cost = 1.1")
            
            # Create common indexes if not exist
            await self._ensure_indexes(conn)
            
            # Prepare common statements
            await self._prepare_common_statements(conn)
    
    async def _ensure_indexes(self, conn: asyncpg.Connection):
        """Ensure performance-critical indexes exist"""
        indexes = [
            ("idx_signals_timestamp", "signals", "created_at DESC"),
            ("idx_signals_symbol_timestamp", "signals", "symbol, created_at DESC"),
            ("idx_orders_status_timestamp", "orders", "status, created_at DESC"),
            ("idx_portfolio_user_timestamp", "portfolio", "user_id, updated_at DESC"),
            ("idx_market_data_symbol_timestamp", "market_data", "symbol, timestamp DESC")
        ]
        
        for index_name, table, columns in indexes:
            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name} 
                    ON {table} ({columns})
                """)
            except Exception as e:
                logger.warning(f"Could not create index {index_name}: {e}")
    
    async def _prepare_common_statements(self, conn: asyncpg.Connection):
        """Prepare commonly used statements"""
        statements = {
            "get_latest_signals": """
                SELECT * FROM signals 
                WHERE symbol = $1 AND created_at > $2 
                ORDER BY created_at DESC 
                LIMIT $3
            """,
            "get_portfolio_value": """
                SELECT SUM(quantity * current_price) as total_value 
                FROM portfolio 
                WHERE user_id = $1 AND active = true
            """,
            "get_order_stats": """
                SELECT status, COUNT(*) as count, AVG(fill_price) as avg_price 
                FROM orders 
                WHERE user_id = $1 AND created_at > $2 
                GROUP BY status
            """
        }
        
        for name, query in statements.items():
            stmt = await conn.prepare(query)
            self.prepared_statements[name] = stmt
    
    @asynccontextmanager
    async def acquire_read(self):
        """Acquire read connection with metrics"""
        start_time = time.time()
        wait_start = time.time()
        
        try:
            async with self.read_pool.acquire() as conn:
                wait_time = time.time() - wait_start
                if wait_time > 1.0:
                    self.metrics["connection_waits"] += 1
                    logger.warning(f"Long connection wait time: {wait_time:.2f}s")
                
                yield conn
                
        except asyncpg.pool.PoolExhaustedError:
            self.metrics["pool_exhausted"] += 1
            logger.error("Read pool exhausted")
            raise
        finally:
            query_time = time.time() - start_time
            self._record_query_time(query_time)
    
    @asynccontextmanager
    async def acquire_write(self):
        """Acquire write connection with metrics"""
        start_time = time.time()
        
        try:
            async with self.write_pool.acquire() as conn:
                yield conn
        except asyncpg.pool.PoolExhaustedError:
            self.metrics["pool_exhausted"] += 1
            logger.error("Write pool exhausted")
            raise
        finally:
            query_time = time.time() - start_time
            self._record_query_time(query_time)
    
    def _record_query_time(self, query_time: float):
        """Record query execution time"""
        self.query_times.append(query_time)
        if len(self.query_times) > 1000:
            self.query_times = self.query_times[-1000:]  # Keep last 1000
        
        self.metrics["total_queries"] += 1
        self.metrics["avg_query_time"] = sum(self.query_times) / len(self.query_times)
        
        if query_time > 1.0:  # Slow query threshold
            self.metrics["slow_queries"] += 1
            logger.warning(f"Slow query detected: {query_time:.2f}s")
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        strategy: QueryOptimizationStrategy = QueryOptimizationStrategy.INDEXED_LOOKUP,
        use_cache: bool = True,
        cache_ttl: int = 300
    ) -> List[Dict[str, Any]]:
        """Execute optimized query with caching"""
        # Generate cache key
        cache_key = None
        if use_cache and self.enable_query_cache:
            cache_key = self._generate_cache_key(query, params)
            
            # Check cache
            if cache_key in self.query_cache:
                cached_result, cached_time = self.query_cache[cache_key]
                if time.time() - cached_time < cache_ttl:
                    self.metrics["cache_hits"] += 1
                    return cached_result
        
        try:
            async with self.acquire_read() as conn:
                # Apply optimization strategy
                optimized_query = self._optimize_query(query, strategy)
                
                # Execute query
                if params:
                    rows = await conn.fetch(optimized_query, *params)
                else:
                    rows = await conn.fetch(optimized_query)
                
                # Convert to dict
                result = [dict(row) for row in rows]
                
                # Cache result
                if cache_key:
                    self.query_cache[cache_key] = (result, time.time())
                
                return result
                
        except Exception as e:
            self.metrics["query_errors"] += 1
            logger.error(f"Query execution error: {e}")
            raise
    
    def _generate_cache_key(self, query: str, params: Optional[Tuple]) -> str:
        """Generate cache key for query"""
        key_parts = [query]
        if params:
            key_parts.extend(str(p) for p in params)
        
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _optimize_query(self, query: str, strategy: QueryOptimizationStrategy) -> str:
        """Apply query optimization strategy"""
        if strategy == QueryOptimizationStrategy.INDEXED_LOOKUP:
            # Ensure index hints for PostgreSQL
            if "WHERE" in query and "/*+ INDEX" not in query:
                # Add index hint comment
                query = query.replace("SELECT", "SELECT /*+ IndexScan */", 1)
        
        elif strategy == QueryOptimizationStrategy.PARTITION_SCAN:
            # Add partition pruning hint
            if "WHERE" in query:
                query = query.replace("SELECT", "SELECT /*+ ENABLE_PARTITION_PRUNING */", 1)
        
        return query
    
    async def batch_insert(
        self,
        table: str,
        columns: List[str],
        values: List[Tuple],
        chunk_size: int = 1000
    ) -> int:
        """Optimized batch insert with chunking"""
        total_inserted = 0
        
        async with self.acquire_write() as conn:
            # Process in chunks to avoid memory issues
            for i in range(0, len(values), chunk_size):
                chunk = values[i:i + chunk_size]
                
                # Use COPY for maximum performance
                result = await conn.copy_records_to_table(
                    table,
                    records=chunk,
                    columns=columns
                )
                
                total_inserted += len(chunk)
        
        logger.info(f"Batch inserted {total_inserted} records into {table}")
        return total_inserted
    
    async def bulk_update(
        self,
        table: str,
        updates: List[Dict[str, Any]],
        key_column: str = "id"
    ) -> int:
        """Optimized bulk update using temporary table"""
        if not updates:
            return 0
        
        async with self.acquire_write() as conn:
            async with conn.transaction():
                # Create temporary table
                temp_table = f"temp_update_{int(time.time() * 1000)}"
                columns = list(updates[0].keys())
                
                await conn.execute(f"""
                    CREATE TEMP TABLE {temp_table} (
                        {', '.join(f'{col} TEXT' for col in columns)}
                    )
                """)
                
                # Insert updates into temp table
                await conn.copy_records_to_table(
                    temp_table,
                    records=[tuple(u.values()) for u in updates],
                    columns=columns
                )
                
                # Perform bulk update
                set_clause = ', '.join(
                    f"{col} = {temp_table}.{col}::TEXT" 
                    for col in columns if col != key_column
                )
                
                result = await conn.execute(f"""
                    UPDATE {table}
                    SET {set_clause}
                    FROM {temp_table}
                    WHERE {table}.{key_column} = {temp_table}.{key_column}::TEXT
                """)
                
                # Extract affected rows count
                affected = int(result.split()[-1])
                
                # Drop temp table
                await conn.execute(f"DROP TABLE {temp_table}")
        
        logger.info(f"Bulk updated {affected} records in {table}")
        return affected
    
    async def create_materialized_view(
        self,
        view_name: str,
        query: str,
        indexes: Optional[List[str]] = None,
        refresh_interval: Optional[int] = None
    ):
        """Create materialized view for complex queries"""
        async with self.acquire_write() as conn:
            # Create materialized view
            await conn.execute(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS {view_name} AS
                {query}
            """)
            
            # Create indexes on view
            if indexes:
                for idx_col in indexes:
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{view_name}_{idx_col}
                        ON {view_name} ({idx_col})
                    """)
            
            # Set up automatic refresh if specified
            if refresh_interval:
                await conn.execute(f"""
                    CREATE OR REPLACE FUNCTION refresh_{view_name}()
                    RETURNS void AS $$
                    BEGIN
                        REFRESH MATERIALIZED VIEW CONCURRENTLY {view_name};
                    END;
                    $$ LANGUAGE plpgsql;
                """)
        
        logger.info(f"Created materialized view: {view_name}")
    
    async def analyze_query_performance(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> Dict[str, Any]:
        """Analyze query performance using EXPLAIN ANALYZE"""
        async with self.acquire_read() as conn:
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
            
            if params:
                result = await conn.fetchval(explain_query, *params)
            else:
                result = await conn.fetchval(explain_query)
            
            return {
                "query": query,
                "execution_plan": result,
                "recommendations": self._generate_optimization_recommendations(result)
            }
    
    def _generate_optimization_recommendations(self, explain_result: Dict) -> List[str]:
        """Generate optimization recommendations from EXPLAIN output"""
        recommendations = []
        
        # This is a simplified version - real implementation would parse the JSON
        # and provide detailed recommendations
        
        plan = explain_result[0]["Plan"]
        
        # Check for sequential scans
        if plan.get("Node Type") == "Seq Scan":
            recommendations.append(f"Consider adding index on {plan.get('Relation Name')}")
        
        # Check for high cost
        if plan.get("Total Cost", 0) > 10000:
            recommendations.append("Query has high cost - consider optimization")
        
        # Check for sort operations
        if "Sort" in str(explain_result):
            recommendations.append("Consider adding index to avoid sorting")
        
        return recommendations
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        metrics = self.metrics.copy()
        
        # Add pool metrics
        if self.read_pool:
            metrics["read_pool"] = {
                "size": self.read_pool.get_size(),
                "free_connections": self.read_pool.get_idle_size(),
                "used_connections": self.read_pool.get_size() - self.read_pool.get_idle_size()
            }
        
        if self.write_pool:
            metrics["write_pool"] = {
                "size": self.write_pool.get_size(),
                "free_connections": self.write_pool.get_idle_size(),
                "used_connections": self.write_pool.get_size() - self.write_pool.get_idle_size()
            }
        
        # Calculate cache metrics
        if self.enable_query_cache:
            total_cache_requests = metrics["cache_hits"] + (metrics["total_queries"] - metrics["cache_hits"])
            metrics["cache_hit_rate"] = (
                metrics["cache_hits"] / total_cache_requests * 100 
                if total_cache_requests > 0 else 0
            )
        
        return metrics 