import time
import functools
import logging
from typing import Dict, Any, Callable
import psutil
import memory_profiler

class PerformanceMonitor:
    """
    Advanced performance monitoring for trading system components.
    Tracks execution time, memory usage, and provides detailed metrics.
    """
    
    def __init__(self, logger=None):
        """
        Initialize performance monitor.
        
        Args:
            logger (logging.Logger, optional): Custom logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = {}
    
    def track_performance(self, func: Callable) -> Callable:
        """
        Decorator to track function performance metrics.
        
        Args:
            func (Callable): Function to monitor
        
        Returns:
            Callable: Wrapped function with performance tracking
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = memory_profiler.memory_usage()[0]
            
            try:
                result = func(*args, **kwargs)
                
                # Compute metrics
                exec_time = time.time() - start_time
                memory_used = memory_profiler.memory_usage()[0] - start_memory
                cpu_percent = psutil.cpu_percent()
                
                # Store metrics
                self.metrics[func.__name__] = {
                    'execution_time': exec_time,
                    'memory_used': memory_used,
                    'cpu_percent': cpu_percent
                }
                
                # Log performance
                self.logger.info(
                    f"Performance: {func.__name__} "
                    f"Time: {exec_time:.4f}s, "
                    f"Memory: {memory_used:.2f}MB, "
                    f"CPU: {cpu_percent}%"
                )
                
                return result
            
            except Exception as e:
                self.logger.error(f"Performance tracking error in {func.__name__}: {e}")
                raise
        
        return wrapper
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance summary.
        
        Returns:
            Dict[str, Any]: Performance metrics summary
        """
        return {
            'total_functions_tracked': len(self.metrics),
            'average_execution_time': sum(
                metric['execution_time'] for metric in self.metrics.values()
            ) / len(self.metrics) if self.metrics else 0,
            'max_memory_usage': max(
                metric['memory_used'] for metric in self.metrics.values()
            ) if self.metrics else 0,
            'detailed_metrics': self.metrics
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
