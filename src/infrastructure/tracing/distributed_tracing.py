"""
Distributed Tracing with OpenTelemetry and Jaeger
Provides comprehensive observability for the platform
"""

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
import logging
from functools import wraps
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class TracingManager:
    """Manages distributed tracing configuration"""
    
    def __init__(self, service_name: str = "goldensignals-ai"):
        self.service_name = service_name
        self.tracer = None
        self.initialized = False
        
    def initialize(self, jaeger_host: str = "localhost", jaeger_port: int = 6831):
        """Initialize OpenTelemetry with Jaeger exporter"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0",
                "deployment.environment": "production"
            })
            
            # Create Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_host,
                agent_port=jaeger_port,
            )
            
            # Create tracer provider
            provider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(jaeger_exporter)
            provider.add_span_processor(processor)
            
            # Set tracer provider
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
            
            # Auto-instrument frameworks
            FastAPIInstrumentor.instrument()
            RedisInstrumentor.instrument()
            SQLAlchemyInstrumentor.instrument()
            
            self.initialized = True
            logger.info(f"Tracing initialized with Jaeger at {jaeger_host}:{jaeger_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            self.initialized = False
    
    def get_tracer(self):
        """Get the configured tracer"""
        if not self.initialized:
            return None
        return self.tracer
    
    def trace_function(self, name: Optional[str] = None):
        """Decorator to trace function execution"""
        def decorator(func):
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.tracer:
                    return await func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(span_name) as span:
                    try:
                        # Add function arguments as span attributes
                        span.set_attribute("function.args", str(args)[:100])
                        span.set_attribute("function.kwargs", str(kwargs)[:100])
                        
                        result = await func(*args, **kwargs)
                        
                        # Add result info
                        span.set_attribute("function.result_type", type(result).__name__)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        
                        return result
                    except Exception as e:
                        # Record exception
                        span.record_exception(e)
                        span.set_status(
                            trace.Status(trace.StatusCode.ERROR, str(e))
                        )
                        raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.tracer:
                    return func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(span_name) as span:
                    try:
                        span.set_attribute("function.args", str(args)[:100])
                        span.set_attribute("function.kwargs", str(kwargs)[:100])
                        
                        result = func(*args, **kwargs)
                        
                        span.set_attribute("function.result_type", type(result).__name__)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(
                            trace.Status(trace.StatusCode.ERROR, str(e))
                        )
                        raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a new span"""
        if not self.tracer:
            return None
        
        span = self.tracer.start_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        return span
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span"""
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(name, attributes or {})

# Global tracing manager
tracing_manager = TracingManager()

# Convenience decorators
def trace_method(name: Optional[str] = None):
    """Decorator for tracing methods"""
    return tracing_manager.trace_function(name)

def trace_agent(agent_type: str):
    """Specialized decorator for agent tracing"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            span_name = f"agent.{agent_type}.{func.__name__}"
            
            if not tracing_manager.tracer:
                return await func(*args, **kwargs)
            
            with tracing_manager.tracer.start_as_current_span(span_name) as span:
                span.set_attribute("agent.type", agent_type)
                span.set_attribute("agent.function", func.__name__)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    if isinstance(result, dict):
                        span.set_attribute("signal.type", result.get("signal", "unknown"))
                        span.set_attribute("signal.confidence", result.get("confidence", 0))
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

def trace_websocket(event_type: str):
    """Specialized decorator for WebSocket tracing"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            span_name = f"websocket.{event_type}"
            
            if not tracing_manager.tracer:
                return await func(*args, **kwargs)
            
            with tracing_manager.tracer.start_as_current_span(span_name) as span:
                span.set_attribute("websocket.event", event_type)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("websocket.success", True)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("websocket.success", False)
                    raise
        
        return wrapper
    return decorator
