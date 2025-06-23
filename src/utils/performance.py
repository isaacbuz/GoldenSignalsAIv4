"""
Performance optimization utilities for GoldenSignalsAI
"""

import asyncio
import time
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import psutil
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PerformanceMonitor:
    """Monitor and optimize performance"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
        
    @contextmanager
    def measure(self, operation: str):
        """Context manager to measure operation time"""
        start_time = time.perf_counter()
        self.start_times[operation] = start_time
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            
            logger.debug(f"{operation} took {duration:.3f}s")
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation"""
        if operation not in self.metrics:
            return {}
            
        times = self.metrics[operation]
        return {
            'count': len(times),
            'total': sum(times),
            'average': np.mean(times),
            'median': np.median(times),
            'min': min(times),
            'max': max(times),
            'std': np.std(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all performance statistics"""
        return {op: self.get_stats(op) for op in self.metrics}
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()


def performance_cache(ttl: int = 60):
    """Cache decorator with TTL"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[str, tuple[T, float]] = {}
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{args}_{kwargs}"
            
            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Update cache
            cache[cache_key] = (result, time.time())
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{args}_{kwargs}"
            
            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Update cache
            cache[cache_key] = (result, time.time())
            
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class AsyncBatchProcessor:
    """Process items in batches asynchronously"""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        items: List[Any],
        processor: Callable[[Any], Any]
    ) -> List[Any]:
        """Process a batch of items"""
        async with self.semaphore:
            tasks = []
            for item in items:
                if asyncio.iscoroutinefunction(processor):
                    task = processor(item)
                else:
                    task = asyncio.get_event_loop().run_in_executor(
                        None, processor, item
                    )
                tasks.append(task)
            
            return await asyncio.gather(*tasks)
    
    async def process_all(
        self,
        items: List[Any],
        processor: Callable[[Any], Any]
    ) -> List[Any]:
        """Process all items in batches"""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await self.process_batch(batch, processor)
            results.extend(batch_results)
        
        return results


class ResourceOptimizer:
    """Optimize resource usage"""
    
    @staticmethod
    def get_optimal_workers() -> int:
        """Get optimal number of workers based on CPU"""
        cpu_count = psutil.cpu_count(logical=True)
        # Use 80% of available CPUs
        return max(1, int(cpu_count * 0.8))
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024**3),  # GB
            'available': memory.available / (1024**3),
            'percent': memory.percent,
            'used': memory.used / (1024**3)
        }
    
    @staticmethod
    def should_use_cache() -> bool:
        """Determine if caching should be used based on available memory"""
        memory = psutil.virtual_memory()
        # Use cache if more than 20% memory available
        return memory.percent < 80
    
    @staticmethod
    def optimize_dataframe(df) -> Any:
        """Optimize pandas DataFrame memory usage"""
        try:
            import pandas as pd
            
            # Downcast numeric columns
            for col in df.select_dtypes(include=['float']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            for col in df.select_dtypes(include=['int']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            # Convert object columns to category if beneficial
            for col in df.select_dtypes(include=['object']).columns:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
            
            return df
            
        except Exception as e:
            logger.warning(f"DataFrame optimization failed: {e}")
            return df


# Global performance monitor instance
_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor"""
    return _monitor


# Convenience functions
def measure_performance(operation: str):
    """Measure performance of an operation"""
    return _monitor.measure(operation)


async def run_parallel(tasks: List[Callable], max_concurrent: int = 10) -> List[Any]:
    """Run tasks in parallel with concurrency limit"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(task):
        async with semaphore:
            if asyncio.iscoroutinefunction(task):
                return await task()
            else:
                return await asyncio.get_event_loop().run_in_executor(None, task)
    
    return await asyncio.gather(*[run_with_semaphore(task) for task in tasks]) 