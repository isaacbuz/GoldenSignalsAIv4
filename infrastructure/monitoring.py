import logging
import time
import traceback
from typing import Any, Dict, Callable
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import sentry_sdk
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from src.infrastructure.config_manager import config_manager

class SystemMonitoring:
    """
    Comprehensive monitoring system for GoldenSignalsAI
    
    Features:
    - Prometheus metrics
    - Distributed tracing
    - Error tracking
    - Performance monitoring
    """
    
    def __init__(self):
        # Sentry error tracking
        sentry_dsn = config_manager.get('monitoring.sentry_dsn', secret=True)
        if sentry_dsn:
            sentry_sdk.init(dsn=sentry_dsn)
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Distributed tracing
        self._setup_distributed_tracing()
    
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Trading performance metrics
        self.trading_signals_total = Counter(
            'goldensignals_trading_signals_total', 
            'Total number of trading signals generated'
        )
        
        self.trading_signal_latency = Histogram(
            'goldensignals_trading_signal_latency_seconds', 
            'Latency of trading signal generation'
        )
        
        self.active_agents = Gauge(
            'goldensignals_active_agents', 
            'Number of active trading agents'
        )
    
    def _setup_distributed_tracing(self):
        """Configure distributed tracing with OpenTelemetry"""
        trace.set_tracer_provider(TracerProvider())
        
        # Jaeger exporter configuration
        jaeger_host = config_manager.get('monitoring.jaeger.host', 'localhost')
        jaeger_port = config_manager.get('monitoring.jaeger.port', 6831)
        
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=jaeger_port
        )
        
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
    
    def trace_function(self, func: Callable) -> Callable:
        """
        Decorator for tracing and monitoring function calls
        
        Args:
            func (Callable): Function to be traced
        
        Returns:
            Callable: Wrapped function with tracing
        """
        tracer = trace.get_tracer(__name__)
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            with tracer.start_as_current_span(func.__name__):
                try:
                    # Increment active agents
                    self.active_agents.inc()
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Record trading signal metrics
                    self.trading_signals_total.inc()
                    self.trading_signal_latency.observe(time.time() - start_time)
                    
                    return result
                
                except Exception as e:
                    # Error tracking
                    sentry_sdk.capture_exception(e)
                    
                    # Log detailed error
                    logging.error(f"Error in {func.__name__}: {e}")
                    logging.error(traceback.format_exc())
                    
                    raise
                
                finally:
                    # Decrement active agents
                    self.active_agents.dec()
        
        return wrapper
    
    def capture_error(self, error: Exception, context: Dict[str, Any] = None):
        """
        Capture and log errors with optional context
        
        Args:
            error (Exception): Error to capture
            context (Dict[str, Any], optional): Additional context
        """
        sentry_sdk.capture_exception(error)
        
        if context:
            sentry_sdk.set_context("trading_context", context)
        
        logging.error(f"Captured error: {error}")
        logging.error(traceback.format_exc())

# Singleton instance
system_monitoring = SystemMonitoring()
