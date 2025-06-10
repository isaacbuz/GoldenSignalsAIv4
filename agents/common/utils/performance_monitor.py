"""
Performance Monitoring Utility for GoldenSignalsAI Agents

Provides essential performance tracking and optimization features.
"""

import asyncio
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque

import numpy as np
from prometheus_client import Histogram, Counter, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
AGENT_EXECUTION_TIME = Histogram(
    'agent_execution_seconds', 
    'Agent execution time', 
    ['agent_name', 'status']
)
AGENT_ERROR_COUNT = Counter(
    'agent_errors_total', 
    'Agent error count', 
    ['agent_name', 'error_type']
)
AGENT_MEMORY_USAGE = Gauge(
    'agent_memory_mb', 
    'Agent memory usage in MB', 
    ['agent_name']
)

@dataclass
class PerformanceSnapshot:
    """Performance measurement snapshot"""
    timestamp: datetime
    agent_name: str
    execution_time: float
    success: bool
    error_type: Optional[str] = None

class PerformanceMonitor:
    """Lightweight performance monitoring for agents"""
    
    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self.snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_snapshots))
        
    @contextmanager
    def monitor_execution(self, agent_name: str):
        """Context manager for monitoring execution"""
        start_time = time.time()
        success = False
        error_type = None
        
        try:
            yield
            success = True
        except Exception as e:
            error_type = type(e).__name__
            AGENT_ERROR_COUNT.labels(agent_name=agent_name, error_type=error_type).inc()
            raise
        finally:
            execution_time = time.time() - start_time
            
            # Update Prometheus metrics
            status = 'success' if success else 'error'
            AGENT_EXECUTION_TIME.labels(agent_name=agent_name, status=status).observe(execution_time)
            
            # Store snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                agent_name=agent_name,
                execution_time=execution_time,
                success=success,
                error_type=error_type
            )
            self.snapshots[agent_name].append(snapshot)
    
    def get_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get performance statistics for an agent"""
        snapshots = list(self.snapshots[agent_name])
        
        if not snapshots:
            return {'error': 'No data available'}
        
        execution_times = [s.execution_time for s in snapshots]
        success_count = sum(1 for s in snapshots if s.success)
        
        return {
            'agent_name': agent_name,
            'total_executions': len(snapshots),
            'success_rate': success_count / len(snapshots),
            'avg_execution_time': np.mean(execution_times),
            'p95_execution_time': np.percentile(execution_times, 95),
            'max_execution_time': np.max(execution_times)
        }

# Global monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(agent_name: str):
    """Decorator for performance monitoring"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with performance_monitor.monitor_execution(agent_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator 