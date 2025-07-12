"""
Performance metrics for GoldenSignalsAI V2.

Provides Prometheus metrics for monitoring system performance.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time

# System metrics
system_info = Info('goldensignals_system', 'System information')
system_info.info({
    'version': '2.0.0',
    'environment': 'production'
})

# Request metrics
http_requests_total = Counter(
    'goldensignals_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'goldensignals_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Signal metrics
signals_generated_total = Counter(
    'goldensignals_signals_generated_total',
    'Total signals generated',
    ['symbol', 'signal_type', 'source']
)

signal_confidence = Histogram(
    'goldensignals_signal_confidence',
    'Signal confidence distribution',
    ['source']
)

# Agent metrics
agent_execution_time = Histogram(
    'goldensignals_agent_execution_seconds',
    'Agent execution time',
    ['agent_name']
)

agent_errors_total = Counter(
    'goldensignals_agent_errors_total',
    'Total agent errors',
    ['agent_name', 'error_type']
)

# Portfolio metrics
portfolio_value = Gauge(
    'goldensignals_portfolio_value_usd',
    'Current portfolio value in USD'
)

portfolio_returns = Gauge(
    'goldensignals_portfolio_returns_percent',
    'Portfolio returns percentage'
)

# Market data metrics
market_data_fetch_duration = Histogram(
    'goldensignals_market_data_fetch_seconds',
    'Market data fetch duration',
    ['provider', 'symbol']
)

market_data_errors = Counter(
    'goldensignals_market_data_errors_total',
    'Market data fetch errors',
    ['provider', 'error_type']
)

# WebSocket metrics
websocket_connections = Gauge(
    'goldensignals_websocket_connections',
    'Active WebSocket connections'
)

websocket_messages_sent = Counter(
    'goldensignals_websocket_messages_sent_total',
    'Total WebSocket messages sent',
    ['message_type']
)

# Database metrics
db_query_duration = Histogram(
    'goldensignals_db_query_duration_seconds',
    'Database query duration',
    ['query_type']
)

db_connection_pool_size = Gauge(
    'goldensignals_db_connection_pool_size',
    'Database connection pool size'
)

# Cache metrics
cache_hits = Counter(
    'goldensignals_cache_hits_total',
    'Cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'goldensignals_cache_misses_total',
    'Cache misses',
    ['cache_type']
)

def track_time(metric: Histogram, **labels):
    """Decorator to track execution time."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                metric.labels(**labels).observe(time.time() - start)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metric.labels(**labels).observe(time.time() - start)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def track_request(method: str, endpoint: str):
    """Decorator to track HTTP requests."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            status = 200
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = getattr(e, 'status_code', 500)
                raise
            finally:
                http_requests_total.labels(method, endpoint, status).inc()
                http_request_duration.labels(method, endpoint).observe(time.time() - start)
        return wrapper
    return decorator

# Business metrics
class BusinessMetrics:
    """Track business-specific metrics."""
    
    @staticmethod
    def track_signal(symbol: str, signal_type: str, source: str, confidence: float):
        """Track signal generation."""
        signals_generated_total.labels(symbol, signal_type, source).inc()
        signal_confidence.labels(source).observe(confidence)
    
    @staticmethod
    def track_portfolio_performance(value: float, returns: float):
        """Track portfolio performance."""
        portfolio_value.set(value)
        portfolio_returns.set(returns)
    
    @staticmethod
    def track_agent_performance(agent_name: str, execution_time: float, error: str = None):
        """Track agent performance."""
        agent_execution_time.labels(agent_name).observe(execution_time)
        if error:
            agent_errors_total.labels(agent_name, error).inc()

import asyncio
