"""
Performance configuration for GoldenSignalsAI
"""

import multiprocessing
from typing import Any, Dict

from src.utils.performance import ResourceOptimizer


class PerformanceConfig:
    """Performance optimization configuration"""

    # CPU Configuration
    MAX_WORKERS = ResourceOptimizer.get_optimal_workers()
    PROCESS_POOL_SIZE = max(1, multiprocessing.cpu_count() // 2)
    THREAD_POOL_SIZE = multiprocessing.cpu_count() * 2

    # Batch Processing
    DEFAULT_BATCH_SIZE = 10
    MAX_CONCURRENT_REQUESTS = 20

    # Caching
    CACHE_TTL = 60  # seconds
    MAX_CACHE_SIZE = 1000  # items
    USE_MEMORY_CACHE = ResourceOptimizer.should_use_cache()

    # Database
    DB_POOL_SIZE = 20
    DB_MAX_OVERFLOW = 10
    DB_POOL_TIMEOUT = 30
    DB_POOL_RECYCLE = 3600

    # API Rate Limiting
    RATE_LIMIT_CALLS = 100
    RATE_LIMIT_PERIOD = 60  # seconds

    # Timeouts
    HTTP_TIMEOUT = 30
    DB_QUERY_TIMEOUT = 10
    CACHE_TIMEOUT = 5

    # Memory Management
    MAX_MEMORY_PERCENT = 80
    DATAFRAME_CHUNK_SIZE = 10000

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get performance configuration as dictionary"""
        return {
            "workers": {
                "max_workers": cls.MAX_WORKERS,
                "process_pool_size": cls.PROCESS_POOL_SIZE,
                "thread_pool_size": cls.THREAD_POOL_SIZE,
            },
            "batch": {
                "default_size": cls.DEFAULT_BATCH_SIZE,
                "max_concurrent": cls.MAX_CONCURRENT_REQUESTS,
            },
            "cache": {
                "ttl": cls.CACHE_TTL,
                "max_size": cls.MAX_CACHE_SIZE,
                "use_memory": cls.USE_MEMORY_CACHE,
            },
            "database": {
                "pool_size": cls.DB_POOL_SIZE,
                "max_overflow": cls.DB_MAX_OVERFLOW,
                "pool_timeout": cls.DB_POOL_TIMEOUT,
                "pool_recycle": cls.DB_POOL_RECYCLE,
            },
            "timeouts": {
                "http": cls.HTTP_TIMEOUT,
                "db_query": cls.DB_QUERY_TIMEOUT,
                "cache": cls.CACHE_TIMEOUT,
            },
            "memory": {
                "max_percent": cls.MAX_MEMORY_PERCENT,
                "chunk_size": cls.DATAFRAME_CHUNK_SIZE,
            },
        }

    @classmethod
    def optimize_for_environment(cls, environment: str = "production"):
        """Optimize configuration based on environment"""
        if environment == "development":
            cls.MAX_WORKERS = 2
            cls.DB_POOL_SIZE = 5
            cls.MAX_CONCURRENT_REQUESTS = 5
        elif environment == "testing":
            cls.MAX_WORKERS = 1
            cls.DB_POOL_SIZE = 2
            cls.MAX_CONCURRENT_REQUESTS = 2
        # Production uses default values
