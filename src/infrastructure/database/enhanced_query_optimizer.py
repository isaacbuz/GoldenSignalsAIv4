"""
Enhanced Database Query Optimizer for GoldenSignalsAI V4
Issue #213: Database Query Optimization
Implements advanced query optimization, monitoring, and performance improvements
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import asyncpg
import redis.asyncio as redis
from asyncpg.pool import Pool
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryPattern(Enum):
    """Common query patterns for optimization"""
    POINT_LOOKUP = "point_lookup"          # Single row by PK
    RANGE_SCAN = "range_scan"              # Date/time ranges
    JOIN_HEAVY = "join_heavy"              # Multiple table joins
    AGGREGATION = "aggregation"            # GROUP BY queries
    FULL_TEXT = "full_text"                # Text search
    BATCH_OPERATION = "batch_operation"    # Bulk insert/update
    REAL_TIME = "real_time"                # Low latency requirements


class IndexSuggestion:
    """Index suggestion based on query analysis"""
    def __init__(self, table: str, columns: List[str], query_pattern: QueryPattern):
        self.table = table
        self.columns = columns
        self.query_pattern = query_pattern
        self.benefit_score = 0.0
        self.estimated_size_mb = 0.0
        
    def to_sql(self) -> str:
        """Generate CREATE INDEX statement"""
        cols = ", ".join(self.columns)
        index_name = f"idx_{self.table}_{'_'.join(self.columns)}"
        return f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} ON {self.table} ({cols})"


class QueryStats:
    """Track query performance statistics"""
    def __init__(self):
        self.execution_count = 0
        self.total_time = 0.0
        self.avg_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.cache_hits = 0
        self.last_execution = None
        self.slow_count = 0
        
    def add_execution(self, exec_time: float, cache_hit: bool = False):
        """Record query execution"""
        self.execution_count += 1
        self.total_time += exec_time
        self.avg_time = self.total_time / self.execution_count
        self.min_time = min(self.min_time, exec_time)
        self.max_time = max(self.max_time, exec_time)
        self.last_execution = datetime.now()
        
        if cache_hit:
            self.cache_hits += 1
        
        if exec_time > 1.0:  # Slow query threshold
            self.slow_count += 1


class EnhancedDatabaseOptimizer:
    """Advanced database optimizer with intelligent caching and query optimization"""
    
    def __init__(
        self,
        database_url: str,
        redis_url: Optional[str] = None,
        enable_query_cache: bool = True,
        enable_auto_indexing: bool = True,
        slow_query_threshold: float = 1.0,
        cache_ttl: int = 300
    ):
        self.database_url = database_url
        self.redis_url = redis_url
        self.enable_query_cache = enable_query_cache
        self.enable_auto_indexing = enable_auto_indexing
        self.slow_query_threshold = slow_query_threshold
        self.cache_ttl = cache_ttl
        
        # Connection pools
        self.read_pool: Optional[Pool] = None
        self.write_pool: Optional[Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Query statistics
        self.query_stats: Dict[str, QueryStats] = defaultdict(QueryStats)
        self.slow_queries: deque = deque(maxlen=1000)
        
        # Index suggestions
        self.index_suggestions: List[IndexSuggestion] = []
        self.applied_indexes: Set[str] = set()
        
        # Prepared statements
        self.prepared_statements: Dict[str, Any] = {}
        
        # Cache invalidation tracking
        self.cache_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
    async def initialize(self):
        """Initialize enhanced database optimizer"""
        try:
            # Create optimized connection pools
            self.read_pool = await self._create_read_pool()
            self.write_pool = await self._create_write_pool()
            
            # Initialize Redis for distributed caching
            if self.redis_url:
                self.redis_client = await redis.from_url(self.redis_url)
            
            # Initialize database optimizations
            await self._initialize_database_optimizations()
            
            # Start background tasks
            asyncio.create_task(self._monitor_query_performance())
            asyncio.create_task(self._auto_index_manager())
            
            logger.info("Enhanced database optimizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            raise
    
    async def _create_read_pool(self) -> Pool:
        """Create optimized read connection pool"""
        return await asyncpg.create_pool(
            self.database_url,
            min_size=20,
            max_size=100,
            max_queries=50000,
            max_inactive_connection_lifetime=300.0,
            command_timeout=60.0,
            statement_cache_size=1000,        # Large statement cache
            max_cached_statement_lifetime=600,
            max_cacheable_statement_size=32768,
            # Connection initialization
            init=self._init_read_connection
        )
    
    async def _create_write_pool(self) -> Pool:
        """Create optimized write connection pool"""
        return await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20,
            command_timeout=120.0,
            statement_cache_size=100,
            # Connection initialization
            init=self._init_write_connection
        )
    
    async def _init_read_connection(self, conn: asyncpg.Connection):
        """Initialize read connection with optimizations"""
        # Set read-optimized parameters
        await conn.execute("SET synchronous_commit = 'off'")
        await conn.execute("SET work_mem = '256MB'")
        await conn.execute("SET effective_cache_size = '4GB'")
        await conn.execute("SET random_page_cost = 1.1")
        await conn.execute("SET cpu_tuple_cost = 0.01")
        
    async def _init_write_connection(self, conn: asyncpg.Connection):
        """Initialize write connection with safety"""
        await conn.execute("SET synchronous_commit = 'on'")
        await conn.execute("SET work_mem = '128MB'")
    
    async def _initialize_database_optimizations(self):
        """Initialize database-level optimizations"""
        async with self.write_pool.acquire() as conn:
            # Create performance monitoring tables if needed
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS query_performance (
                    id SERIAL PRIMARY KEY,
                    query_hash VARCHAR(64),
                    query_text TEXT,
                    execution_time FLOAT,
                    rows_returned INT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_performance_hash 
                ON query_performance (query_hash, timestamp DESC)
            """)
            
            # Create suggested indexes table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS suggested_indexes (
                    id SERIAL PRIMARY KEY,
                    table_name VARCHAR(255),
                    columns TEXT[],
                    benefit_score FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    applied BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Load existing indexes
            rows = await conn.fetch("""
                SELECT indexname FROM pg_indexes 
                WHERE schemaname = 'public'
            """)
            self.applied_indexes.update(row['indexname'] for row in rows)
    
    @asynccontextmanager
    async def smart_query(self, query_pattern: QueryPattern = QueryPattern.POINT_LOOKUP):
        """Context manager for smart query execution"""
        start_time = time.time()
        conn = None
        
        try:
            # Choose pool based on query pattern
            if query_pattern in [QueryPattern.REAL_TIME, QueryPattern.POINT_LOOKUP]:
                conn = await self.read_pool.acquire()
            else:
                # Use different connection for heavy queries
                conn = await self.read_pool.acquire()
                await conn.execute("SET statement_timeout = '5min'")
            
            yield conn
            
        finally:
            if conn:
                await self.read_pool.release(conn)
            
            # Record execution time
            exec_time = time.time() - start_time
            if exec_time > self.slow_query_threshold:
                logger.warning(f"Slow query detected: {exec_time:.2f}s")
    
    async def execute_optimized(
        self,
        query: str,
        params: Optional[Tuple] = None,
        query_pattern: QueryPattern = QueryPattern.POINT_LOOKUP,
        use_cache: bool = True,
        invalidate_tables: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Execute query with optimizations"""
        # Generate query hash
        query_hash = self._hash_query(query, params)
        
        # Check cache
        if use_cache and self.enable_query_cache:
            cached = await self._get_cached_result(query_hash)
            if cached is not None:
                self.query_stats[query_hash].add_execution(0.0, cache_hit=True)
                return cached
        
        # Execute query
        start_time = time.time()
        
        async with self.smart_query(query_pattern) as conn:
            # Use prepared statement if available
            if query_hash in self.prepared_statements:
                stmt = self.prepared_statements[query_hash]
                rows = await stmt.fetch(*params) if params else await stmt.fetch()
            else:
                # Prepare and cache statement for future use
                stmt = await conn.prepare(query)
                self.prepared_statements[query_hash] = stmt
                rows = await stmt.fetch(*params) if params else await stmt.fetch()
            
            result = [dict(row) for row in rows]
        
        exec_time = time.time() - start_time
        
        # Update statistics
        self.query_stats[query_hash].add_execution(exec_time)
        
        # Cache result
        if use_cache and self.enable_query_cache:
            await self._cache_result(query_hash, result, invalidate_tables)
        
        # Log slow query
        if exec_time > self.slow_query_threshold:
            await self._log_slow_query(query, params, exec_time, len(result))
        
        # Analyze for optimization opportunities
        if self.enable_auto_indexing:
            await self._analyze_query_for_optimization(query, exec_time)
        
        return result
    
    async def batch_insert_optimized(
        self,
        table: str,
        records: List[Dict[str, Any]],
        chunk_size: int = 10000,
        on_conflict: Optional[str] = None
    ) -> int:
        """Optimized batch insert with conflict handling"""
        if not records:
            return 0
        
        # Prepare data
        columns = list(records[0].keys())
        total_inserted = 0
        
        async with self.write_pool.acquire() as conn:
            # Use transaction for consistency
            async with conn.transaction():
                # Process in chunks
                for i in range(0, len(records), chunk_size):
                    chunk = records[i:i + chunk_size]
                    
                    # Convert to tuples
                    values = [tuple(r[col] for col in columns) for r in chunk]
                    
                    if on_conflict:
                        # Use INSERT ... ON CONFLICT
                        placeholders = ', '.join(f'${i+1}' for i in range(len(columns)))
                        values_clause = ', '.join(
                            f'({placeholders})' for _ in chunk
                        )
                        
                        query = f"""
                            INSERT INTO {table} ({', '.join(columns)})
                            VALUES {values_clause}
                            {on_conflict}
                        """
                        
                        # Flatten values
                        flat_values = [v for row in values for v in row]
                        await conn.execute(query, *flat_values)
                    else:
                        # Use COPY for maximum performance
                        await conn.copy_records_to_table(
                            table,
                            records=values,
                            columns=columns
                        )
                    
                    total_inserted += len(chunk)
        
        # Invalidate related caches
        await self._invalidate_table_cache(table)
        
        logger.info(f"Batch inserted {total_inserted} records into {table}")
        return total_inserted
    
    async def create_smart_index(self, suggestion: IndexSuggestion) -> bool:
        """Create index with minimal impact"""
        try:
            async with self.write_pool.acquire() as conn:
                # Create index concurrently to avoid locking
                await conn.execute(suggestion.to_sql())
                
                # Update tracking
                self.applied_indexes.add(f"idx_{suggestion.table}_{'_'.join(suggestion.columns)}")
                
                # Mark as applied in suggestions table
                await conn.execute("""
                    UPDATE suggested_indexes 
                    SET applied = TRUE 
                    WHERE table_name = $1 AND columns = $2
                """, suggestion.table, suggestion.columns)
                
                logger.info(f"Created index on {suggestion.table}({', '.join(suggestion.columns)})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    async def analyze_table_statistics(self, table: str) -> Dict[str, Any]:
        """Analyze table statistics for optimization"""
        async with self.read_pool.acquire() as conn:
            # Update table statistics
            await conn.execute(f"ANALYZE {table}")
            
            # Get table size and stats
            stats = await conn.fetchrow("""
                SELECT 
                    pg_size_pretty(pg_total_relation_size($1)) as total_size,
                    pg_size_pretty(pg_relation_size($1)) as table_size,
                    pg_size_pretty(pg_indexes_size($1)) as indexes_size,
                    n_live_tup as row_count,
                    n_dead_tup as dead_rows,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze
                FROM pg_stat_user_tables
                WHERE relname = $1
            """, table)
            
            # Get index usage
            index_stats = await conn.fetch("""
                SELECT 
                    indexrelname as index_name,
                    idx_scan as scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched,
                    pg_size_pretty(pg_relation_size(indexrelid)) as size
                FROM pg_stat_user_indexes
                WHERE relname = $1
                ORDER BY idx_scan DESC
            """, table)
            
            return {
                "table": table,
                "statistics": dict(stats) if stats else {},
                "indexes": [dict(idx) for idx in index_stats],
                "recommendations": self._generate_table_recommendations(stats, index_stats)
            }
    
    def _generate_table_recommendations(self, stats: Any, index_stats: List[Any]) -> List[str]:
        """Generate optimization recommendations for table"""
        recommendations = []
        
        if stats:
            # Check for bloat
            if stats['dead_rows'] > stats['row_count'] * 0.2:
                recommendations.append(f"High dead tuple ratio - consider VACUUM")
            
            # Check for missing statistics
            if not stats['last_analyze']:
                recommendations.append("Table never analyzed - run ANALYZE")
            
            # Check unused indexes
            for idx in index_stats:
                if idx['scans'] == 0:
                    recommendations.append(f"Unused index {idx['index_name']} - consider dropping")
        
        return recommendations
    
    async def _monitor_query_performance(self):
        """Background task to monitor query performance"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Analyze slow queries
                if self.slow_queries:
                    await self._analyze_slow_queries()
                
                # Generate index suggestions
                if self.enable_auto_indexing:
                    await self._generate_index_suggestions()
                
                # Clean up old prepared statements
                await self._cleanup_prepared_statements()
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
    
    async def _auto_index_manager(self):
        """Background task to automatically create beneficial indexes"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                if not self.enable_auto_indexing:
                    continue
                
                # Apply high-benefit index suggestions
                for suggestion in self.index_suggestions:
                    if suggestion.benefit_score > 0.8:  # High benefit threshold
                        index_name = f"idx_{suggestion.table}_{'_'.join(suggestion.columns)}"
                        if index_name not in self.applied_indexes:
                            await self.create_smart_index(suggestion)
                
            except Exception as e:
                logger.error(f"Error in auto index manager: {e}")
    
    def _hash_query(self, query: str, params: Optional[Tuple]) -> str:
        """Generate hash for query and parameters"""
        key = query
        if params:
            key += "|" + "|".join(str(p) for p in params)
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def _get_cached_result(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached query result"""
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"query:{query_hash}")
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None
    
    async def _cache_result(
        self, 
        query_hash: str, 
        result: List[Dict[str, Any]],
        tables: Optional[List[str]] = None
    ):
        """Cache query result with dependency tracking"""
        if self.redis_client:
            try:
                # Cache result
                await self.redis_client.setex(
                    f"query:{query_hash}",
                    self.cache_ttl,
                    json.dumps(result, default=str)
                )
                
                # Track dependencies
                if tables:
                    for table in tables:
                        self.cache_dependencies[table].add(query_hash)
                        
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
    
    async def _invalidate_table_cache(self, table: str):
        """Invalidate all cached queries for a table"""
        if table in self.cache_dependencies:
            for query_hash in self.cache_dependencies[table]:
                if self.redis_client:
                    await self.redis_client.delete(f"query:{query_hash}")
            self.cache_dependencies[table].clear()
    
    async def _log_slow_query(
        self, 
        query: str, 
        params: Optional[Tuple], 
        exec_time: float,
        rows_returned: int
    ):
        """Log slow query for analysis"""
        self.slow_queries.append({
            'query': query,
            'params': params,
            'execution_time': exec_time,
            'rows_returned': rows_returned,
            'timestamp': datetime.now()
        })
        
        # Store in database for persistence
        async with self.write_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO query_performance (query_hash, query_text, execution_time, rows_returned)
                VALUES ($1, $2, $3, $4)
            """, self._hash_query(query, params), query, exec_time, rows_returned)
    
    async def _analyze_query_for_optimization(self, query: str, exec_time: float):
        """Analyze query for optimization opportunities"""
        if exec_time < self.slow_query_threshold:
            return
        
        # Simple pattern matching for index suggestions
        # In production, use EXPLAIN output for better analysis
        
        if "WHERE" in query:
            # Extract WHERE conditions
            where_clause = query.split("WHERE")[1].split("ORDER BY")[0]
            
            # Look for column references
            for table_col in where_clause.split("AND"):
                if "." in table_col and "=" in table_col:
                    parts = table_col.strip().split(".")
                    if len(parts) == 2:
                        table = parts[0].strip()
                        col = parts[1].split("=")[0].strip()
                        
                        # Create index suggestion
                        suggestion = IndexSuggestion(
                            table=table,
                            columns=[col],
                            query_pattern=QueryPattern.POINT_LOOKUP
                        )
                        suggestion.benefit_score = min(1.0, exec_time / 10.0)
                        
                        self.index_suggestions.append(suggestion)
    
    async def _analyze_slow_queries(self):
        """Analyze slow queries for patterns"""
        # Group by query pattern
        pattern_groups = defaultdict(list)
        
        for sq in self.slow_queries:
            # Simple pattern detection
            query = sq['query'].upper()
            if "JOIN" in query:
                pattern = QueryPattern.JOIN_HEAVY
            elif "GROUP BY" in query:
                pattern = QueryPattern.AGGREGATION
            elif "WHERE" in query and "BETWEEN" in query:
                pattern = QueryPattern.RANGE_SCAN
            else:
                pattern = QueryPattern.POINT_LOOKUP
            
            pattern_groups[pattern].append(sq)
    
    async def _generate_index_suggestions(self):
        """Generate index suggestions from query patterns"""
        # Analyze recent slow queries
        for sq in list(self.slow_queries)[-100:]:
            await self._analyze_query_for_optimization(sq['query'], sq['execution_time'])
    
    async def _cleanup_prepared_statements(self):
        """Clean up old prepared statements"""
        # Keep only frequently used statements
        threshold = datetime.now() - timedelta(hours=1)
        
        to_remove = []
        for query_hash, stats in self.query_stats.items():
            if stats.last_execution and stats.last_execution < threshold:
                if stats.execution_count < 10:  # Low usage
                    to_remove.append(query_hash)
        
        for query_hash in to_remove:
            self.prepared_statements.pop(query_hash, None)
            self.query_stats.pop(query_hash, None)
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_queries": sum(s.execution_count for s in self.query_stats.values()),
                "unique_queries": len(self.query_stats),
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "avg_query_time": self._calculate_avg_query_time(),
                "slow_queries": len(self.slow_queries)
            },
            "top_slow_queries": list(self.slow_queries)[-10:],
            "index_suggestions": [
                {
                    "table": s.table,
                    "columns": s.columns,
                    "benefit_score": s.benefit_score,
                    "sql": s.to_sql()
                }
                for s in sorted(self.index_suggestions, key=lambda x: x.benefit_score, reverse=True)[:10]
            ],
            "connection_pool_status": {
                "read_pool": {
                    "size": self.read_pool.get_size() if self.read_pool else 0,
                    "idle": self.read_pool.get_idle_size() if self.read_pool else 0
                },
                "write_pool": {
                    "size": self.write_pool.get_size() if self.write_pool else 0,
                    "idle": self.write_pool.get_idle_size() if self.write_pool else 0
                }
            }
        }
        
        return report
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        total_hits = sum(s.cache_hits for s in self.query_stats.values())
        total_executions = sum(s.execution_count for s in self.query_stats.values())
        return total_hits / total_executions if total_executions > 0 else 0.0
    
    def _calculate_avg_query_time(self) -> float:
        """Calculate average query execution time"""
        total_time = sum(s.total_time for s in self.query_stats.values())
        total_count = sum(s.execution_count for s in self.query_stats.values())
        return total_time / total_count if total_count > 0 else 0.0
    
    async def close(self):
        """Clean shutdown"""
        if self.read_pool:
            await self.read_pool.close()
        if self.write_pool:
            await self.write_pool.close()
        if self.redis_client:
            await self.redis_client.close()


# Demo function
async def demo_enhanced_optimizer():
    """Demonstrate enhanced database optimizer capabilities"""
    print("Enhanced Database Query Optimizer Demo - Issue #213")
    print("="*70)
    
    # Initialize optimizer
    optimizer = EnhancedDatabaseOptimizer(
        database_url="postgresql://user:pass@localhost/goldensignals",
        redis_url="redis://localhost:6379",
        enable_auto_indexing=True
    )
    
    await optimizer.initialize()
    
    print("\n📊 Query Optimization Examples")
    print("-"*50)
    
    # Example 1: Simple point lookup with caching
    print("\n1️⃣ Point Lookup Query (with caching)")
    
    result = await optimizer.execute_optimized(
        query="SELECT * FROM signals WHERE symbol = $1 AND created_at > $2 LIMIT 10",
        params=("AAPL", datetime.now() - timedelta(hours=1)),
        query_pattern=QueryPattern.POINT_LOOKUP
    )
    print(f"   First execution: {len(result)} results")
    
    # Second execution should hit cache
    result2 = await optimizer.execute_optimized(
        query="SELECT * FROM signals WHERE symbol = $1 AND created_at > $2 LIMIT 10",
        params=("AAPL", datetime.now() - timedelta(hours=1)),
        query_pattern=QueryPattern.POINT_LOOKUP
    )
    print(f"   Cached execution: {len(result2)} results (from cache)")
    
    # Example 2: Batch insert optimization
    print("\n2️⃣ Optimized Batch Insert")
    
    records = [
        {
            "symbol": f"TEST{i}",
            "signal_type": "BUY",
            "confidence": 0.85,
            "price": 100.0 + i,
            "created_at": datetime.now()
        }
        for i in range(10000)
    ]
    
    start_time = time.time()
    inserted = await optimizer.batch_insert_optimized(
        table="signals",
        records=records,
        on_conflict="ON CONFLICT (symbol, created_at) DO NOTHING"
    )
    insert_time = time.time() - start_time
    
    print(f"   Inserted {inserted} records in {insert_time:.2f}s")
    print(f"   Rate: {inserted/insert_time:.0f} records/second")
    
    # Example 3: Complex aggregation query
    print("\n3️⃣ Complex Aggregation Query")
    
    agg_result = await optimizer.execute_optimized(
        query="""
            SELECT 
                symbol,
                DATE_TRUNC('hour', created_at) as hour,
                COUNT(*) as signal_count,
                AVG(confidence) as avg_confidence,
                MAX(price) as max_price
            FROM signals
            WHERE created_at > $1
            GROUP BY symbol, hour
            ORDER BY signal_count DESC
            LIMIT 20
        """,
        params=(datetime.now() - timedelta(days=1),),
        query_pattern=QueryPattern.AGGREGATION
    )
    
    print(f"   Aggregated {len(agg_result)} symbol-hour combinations")
    
    # Example 4: Table statistics and recommendations
    print("\n4️⃣ Table Analysis and Recommendations")
    
    table_stats = await optimizer.analyze_table_statistics("signals")
    
    print(f"\n   Table: {table_stats['table']}")
    stats = table_stats['statistics']
    if stats:
        print(f"   Size: {stats.get('total_size', 'N/A')}")
        print(f"   Rows: {stats.get('row_count', 'N/A'):,}")
        print(f"   Dead rows: {stats.get('dead_rows', 'N/A'):,}")
    
    print("\n   Indexes:")
    for idx in table_stats['indexes'][:3]:
        print(f"   - {idx['index_name']}: {idx['scans']} scans, {idx['size']}")
    
    if table_stats['recommendations']:
        print("\n   Recommendations:")
        for rec in table_stats['recommendations']:
            print(f"   ⚠️ {rec}")
    
    # Example 5: Optimization report
    print("\n5️⃣ Optimization Report")
    
    report = await optimizer.get_optimization_report()
    
    print(f"\n   Statistics:")
    print(f"   - Total queries: {report['statistics']['total_queries']}")
    print(f"   - Cache hit rate: {report['statistics']['cache_hit_rate']:.1%}")
    print(f"   - Avg query time: {report['statistics']['avg_query_time']:.3f}s")
    print(f"   - Slow queries: {report['statistics']['slow_queries']}")
    
    if report['index_suggestions']:
        print(f"\n   Top Index Suggestions:")
        for suggestion in report['index_suggestions'][:3]:
            print(f"   - {suggestion['table']}({', '.join(suggestion['columns'])})")
            print(f"     Benefit score: {suggestion['benefit_score']:.2f}")
    
    print(f"\n   Connection Pools:")
    print(f"   - Read pool: {report['connection_pool_status']['read_pool']['idle']}/{report['connection_pool_status']['read_pool']['size']} idle")
    print(f"   - Write pool: {report['connection_pool_status']['write_pool']['idle']}/{report['connection_pool_status']['write_pool']['size']} idle")
    
    # Cleanup
    await optimizer.close()
    
    print("\n" + "="*70)
    print("✅ Enhanced Database Optimizer demonstrates:")
    print("- Intelligent query caching with Redis")
    print("- Automatic index suggestions")
    print("- Optimized batch operations")
    print("- Query performance monitoring")
    print("- Connection pool optimization")
    print("- Table statistics and recommendations")


if __name__ == "__main__":
    asyncio.run(demo_enhanced_optimizer()) 