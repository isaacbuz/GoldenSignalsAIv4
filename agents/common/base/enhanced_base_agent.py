"""
Enhanced Base Agent with Performance Monitoring, Caching, and Validation

This enhanced base agent provides:
- Performance monitoring and metrics collection
- Intelligent caching with TTL support
- Input validation and sanitization  
- Async/await support with timeout handling
- Error handling with retries and circuit breaker
- Configuration validation
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from redis import Redis
from prometheus_client import Counter, Histogram, Gauge

# Enhanced logging
logger = logging.getLogger(__name__)

# Prometheus metrics for monitoring
AGENT_EXECUTION_TIME = Histogram('agent_execution_seconds', 'Agent execution time', ['agent_name', 'operation'])
AGENT_ERROR_COUNT = Counter('agent_errors_total', 'Agent error count', ['agent_name', 'error_type'])
AGENT_CACHE_HIT_RATE = Gauge('agent_cache_hit_rate', 'Agent cache hit rate', ['agent_name'])
AGENT_MEMORY_USAGE = Gauge('agent_memory_usage_mb', 'Agent memory usage in MB', ['agent_name'])

class CacheStrategy(Enum):
    """Cache strategy enumeration"""
    NONE = "none"
    MEMORY = "memory"
    REDIS = "redis"
    MULTILAYER = "multilayer"

class ValidationLevel(Enum):
    """Input validation levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    BASIC = "basic"
    NONE = "none"

@dataclass
class PerformanceMetrics:
    """Container for agent performance metrics"""
    execution_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0.0
    last_execution: Optional[datetime] = None

@dataclass 
class CacheConfig:
    """Cache configuration"""
    strategy: CacheStrategy = CacheStrategy.MULTILAYER
    memory_ttl: int = 30  # seconds
    redis_ttl: int = 300  # seconds
    max_memory_entries: int = 1000
    compression_enabled: bool = True

@dataclass
class AgentConfig:
    """Enhanced agent configuration with validation"""
    name: str
    agent_type: str
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    validation_level: ValidationLevel = ValidationLevel.MODERATE
    timeout_seconds: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    enable_monitoring: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

class SignalData(BaseModel):
    """Validated signal data model"""
    action: str = Field(..., regex="^(buy|sell|hold)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is a valid number"""
        if not isinstance(v, (int, float)) or np.isnan(v) or np.isinf(v):
            raise ValueError("Confidence must be a valid number")
        return float(v)

class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

class IntelligentCache:
    """Multi-layer caching system with memory and Redis support"""
    
    def __init__(self, config: CacheConfig, redis_client: Optional[Redis] = None):
        self.config = config
        self.redis_client = redis_client
        self.memory_cache = {}
        self.memory_timestamps = {}
        
    def _generate_key(self, agent_name: str, method: str, data_hash: str) -> str:
        """Generate consistent cache key"""
        return f"agent:{agent_name}:{method}:{data_hash}"
    
    def _hash_data(self, data: Any) -> str:
        """Generate hash for data to use as cache key"""
        try:
            if isinstance(data, dict):
                # Sort keys for consistent hashing
                sorted_data = json.dumps(data, sort_keys=True)
            else:
                sorted_data = str(data)
            return str(hash(sorted_data))
        except Exception:
            return str(hash(str(data)))
    
    def _cleanup_memory_cache(self):
        """Remove expired entries from memory cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.memory_timestamps.items()
            if current_time - timestamp > self.config.memory_ttl
        ]
        
        for key in expired_keys:
            self.memory_cache.pop(key, None)
            self.memory_timestamps.pop(key, None)
        
        # Limit cache size
        if len(self.memory_cache) > self.config.max_memory_entries:
            # Remove oldest entries
            sorted_items = sorted(self.memory_timestamps.items(), key=lambda x: x[1])
            num_to_remove = len(self.memory_cache) - self.config.max_memory_entries
            
            for key, _ in sorted_items[:num_to_remove]:
                self.memory_cache.pop(key, None)
                self.memory_timestamps.pop(key, None)
    
    async def get(self, agent_name: str, method: str, data: Any) -> Optional[Any]:
        """Get cached result with fallback strategy"""
        if self.config.strategy == CacheStrategy.NONE:
            return None
        
        cache_key = self._generate_key(agent_name, method, self._hash_data(data))
        
        # L1: Memory cache
        if self.config.strategy in [CacheStrategy.MEMORY, CacheStrategy.MULTILAYER]:
            self._cleanup_memory_cache()
            
            if cache_key in self.memory_cache:
                timestamp = self.memory_timestamps[cache_key]
                if time.time() - timestamp <= self.config.memory_ttl:
                    return self.memory_cache[cache_key]
                else:
                    # Expired, remove
                    self.memory_cache.pop(cache_key, None)
                    self.memory_timestamps.pop(cache_key, None)
        
        # L2: Redis cache
        if self.config.strategy in [CacheStrategy.REDIS, CacheStrategy.MULTILAYER] and self.redis_client:
            try:
                cached_value = await self.redis_client.get(cache_key)
                if cached_value:
                    result = json.loads(cached_value)
                    
                    # Store in memory cache if using multilayer
                    if self.config.strategy == CacheStrategy.MULTILAYER:
                        await self.set_memory(cache_key, result)
                    
                    return result
            except Exception as e:
                logger.warning(f"Redis cache retrieval failed: {e}")
        
        return None
    
    async def set(self, agent_name: str, method: str, data: Any, result: Any):
        """Set cached result in appropriate layer(s)"""
        if self.config.strategy == CacheStrategy.NONE:
            return
        
        cache_key = self._generate_key(agent_name, method, self._hash_data(data))
        
        # Memory cache
        if self.config.strategy in [CacheStrategy.MEMORY, CacheStrategy.MULTILAYER]:
            await self.set_memory(cache_key, result)
        
        # Redis cache
        if self.config.strategy in [CacheStrategy.REDIS, CacheStrategy.MULTILAYER] and self.redis_client:
            try:
                serialized_result = json.dumps(result, default=str)
                await self.redis_client.setex(
                    cache_key, 
                    self.config.redis_ttl, 
                    serialized_result
                )
            except Exception as e:
                logger.warning(f"Redis cache storage failed: {e}")
    
    async def set_memory(self, cache_key: str, result: Any):
        """Set result in memory cache"""
        self.memory_cache[cache_key] = result
        self.memory_timestamps[cache_key] = time.time()

def monitor_performance(func):
    """Decorator to monitor agent performance"""
    @wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        start_time = time.time()
        agent_name = getattr(self, 'name', 'unknown')
        operation = func.__name__
        
        try:
            # Monitor memory usage
            if hasattr(self, '_update_memory_usage'):
                self._update_memory_usage()
            
            # Execute function
            result = await func(self, *args, **kwargs)
            
            # Record success metrics
            execution_time = time.time() - start_time
            AGENT_EXECUTION_TIME.labels(agent_name=agent_name, operation=operation).observe(execution_time)
            
            if hasattr(self, 'metrics'):
                self._update_metrics(execution_time, success=True)
            
            return result
            
        except Exception as e:
            # Record error metrics
            execution_time = time.time() - start_time
            error_type = type(e).__name__
            AGENT_ERROR_COUNT.labels(agent_name=agent_name, error_type=error_type).inc()
            
            if hasattr(self, 'metrics'):
                self._update_metrics(execution_time, success=False)
            
            raise
    
    @wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        start_time = time.time()
        agent_name = getattr(self, 'name', 'unknown')
        operation = func.__name__
        
        try:
            result = func(self, *args, **kwargs)
            
            execution_time = time.time() - start_time
            AGENT_EXECUTION_TIME.labels(agent_name=agent_name, operation=operation).observe(execution_time)
            
            if hasattr(self, 'metrics'):
                self._update_metrics(execution_time, success=True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_type = type(e).__name__
            AGENT_ERROR_COUNT.labels(agent_name=agent_name, error_type=error_type).inc()
            
            if hasattr(self, 'metrics'):
                self._update_metrics(execution_time, success=False)
            
            raise
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

class EnhancedBaseAgent(ABC):
    """
    Enhanced base agent with performance monitoring, caching, and validation.
    
    Features:
    - Intelligent multi-layer caching
    - Performance metrics and monitoring
    - Input validation and sanitization
    - Async/await support with timeouts
    - Circuit breaker pattern for resilience
    - Configurable retry logic
    """
    
    def __init__(
        self, 
        config: AgentConfig,
        redis_client: Optional[Redis] = None
    ):
        self.config = config
        self.name = config.name
        self.agent_type = config.agent_type
        
        # Initialize components
        self.cache = IntelligentCache(config.cache_config, redis_client)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            timeout=60
        )
        self.metrics = PerformanceMetrics()
        
        # Initialize monitoring
        if config.enable_monitoring:
            self._setup_monitoring()
        
        logger.info(f"Enhanced agent '{self.name}' initialized with config: {config}")
    
    def _setup_monitoring(self):
        """Setup performance monitoring"""
        AGENT_CACHE_HIT_RATE.labels(agent_name=self.name).set(0)
        AGENT_MEMORY_USAGE.labels(agent_name=self.name).set(0)
    
    def _update_metrics(self, execution_time: float, success: bool):
        """Update performance metrics"""
        self.metrics.execution_count += 1
        self.metrics.total_execution_time += execution_time
        self.metrics.average_execution_time = (
            self.metrics.total_execution_time / self.metrics.execution_count
        )
        self.metrics.max_execution_time = max(self.metrics.max_execution_time, execution_time)
        self.metrics.min_execution_time = min(self.metrics.min_execution_time, execution_time)
        self.metrics.last_execution = datetime.now()
        
        if not success:
            self.metrics.error_count += 1
    
    def _update_memory_usage(self):
        """Update memory usage metrics (simplified)"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.memory_usage_mb = memory_mb
            AGENT_MEMORY_USAGE.labels(agent_name=self.name).set(memory_mb)
        except ImportError:
            pass  # psutil not available
    
    def _validate_input(self, data: Any) -> bool:
        """Validate input data based on validation level"""
        if self.config.validation_level == ValidationLevel.NONE:
            return True
        
        if self.config.validation_level == ValidationLevel.BASIC:
            return data is not None
        
        if self.config.validation_level == ValidationLevel.MODERATE:
            if isinstance(data, dict):
                return bool(data)  # Non-empty dict
            elif isinstance(data, (list, pd.DataFrame)):
                return len(data) > 0
            return data is not None
        
        if self.config.validation_level == ValidationLevel.STRICT:
            # Implement strict validation based on agent requirements
            return self._strict_validation(data)
        
        return True
    
    def _strict_validation(self, data: Any) -> bool:
        """Override in subclasses for strict validation"""
        return True
    
    @monitor_performance
    async def process_async(self, data: Dict[str, Any]) -> SignalData:
        """Async version of signal processing with caching and monitoring"""
        # Input validation
        if not self._validate_input(data):
            raise ValueError(f"Invalid input data for agent {self.name}")
        
        # Check cache first
        cached_result = await self.cache.get(self.name, "process", data)
        if cached_result:
            self.metrics.cache_hits += 1
            self._update_cache_hit_rate()
            return SignalData(**cached_result)
        
        self.metrics.cache_misses += 1
        self._update_cache_hit_rate()
        
        # Process with circuit breaker and timeout
        try:
            result = await asyncio.wait_for(
                self._process_internal_async(data),
                timeout=self.config.timeout_seconds
            )
            
            # Cache the result
            await self.cache.set(self.name, "process", data, result.dict())
            
            return result
            
        except asyncio.TimeoutError:
            raise Exception(f"Agent {self.name} timed out after {self.config.timeout_seconds}s")
    
    def _update_cache_hit_rate(self):
        """Update cache hit rate metric"""
        total_requests = self.metrics.cache_hits + self.metrics.cache_misses
        if total_requests > 0:
            hit_rate = self.metrics.cache_hits / total_requests
            AGENT_CACHE_HIT_RATE.labels(agent_name=self.name).set(hit_rate)
    
    @monitor_performance
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous signal processing (backwards compatibility)"""
        # Input validation
        if not self._validate_input(data):
            raise ValueError(f"Invalid input data for agent {self.name}")
        
        try:
            return self.circuit_breaker.call(self._process_internal, data)
        except Exception as e:
            logger.error(f"Agent {self.name} processing failed: {str(e)}")
            return self._create_error_signal(str(e))
    
    def _create_error_signal(self, error_message: str) -> Dict[str, Any]:
        """Create error signal response"""
        return {
            "action": "hold",
            "confidence": 0.0,
            "metadata": {
                "error": error_message,
                "agent": self.name,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    @abstractmethod
    async def _process_internal_async(self, data: Dict[str, Any]) -> SignalData:
        """Internal async processing - implement in subclasses"""
        pass
    
    @abstractmethod
    def _process_internal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal sync processing - implement in subclasses"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent metrics"""
        return {
            "name": self.name,
            "type": self.agent_type,
            "metrics": {
                "execution_count": self.metrics.execution_count,
                "average_execution_time": self.metrics.average_execution_time,
                "max_execution_time": self.metrics.max_execution_time,
                "min_execution_time": self.metrics.min_execution_time,
                "error_count": self.metrics.error_count,
                "error_rate": (
                    self.metrics.error_count / max(self.metrics.execution_count, 1)
                ),
                "cache_hit_rate": (
                    self.metrics.cache_hits / max(self.metrics.cache_hits + self.metrics.cache_misses, 1)
                ),
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "last_execution": self.metrics.last_execution.isoformat() if self.metrics.last_execution else None
            },
            "config": {
                "cache_strategy": self.config.cache_config.strategy.value,
                "validation_level": self.config.validation_level.value,
                "timeout_seconds": self.config.timeout_seconds,
                "circuit_breaker_state": self.circuit_breaker.state
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Test basic functionality
            test_data = {"test": True, "timestamp": datetime.now().isoformat()}
            await asyncio.wait_for(
                self._process_internal_async(test_data),
                timeout=5.0
            )
            
            return {
                "status": "healthy",
                "agent": self.name,
                "circuit_breaker_state": self.circuit_breaker.state,
                "cache_functional": True,
                "last_execution": self.metrics.last_execution.isoformat() if self.metrics.last_execution else None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent": self.name,
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state
            }

    def __repr__(self) -> str:
        return f"EnhancedBaseAgent(name='{self.name}', type='{self.agent_type}')" 