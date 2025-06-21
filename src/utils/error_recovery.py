"""
Error Recovery System for GoldenSignalsAI
Implements circuit breakers, retry logic, and fallback mechanisms
"""

import asyncio
import functools
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from enum import Enum
import traceback
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ErrorContext:
    """Context information for error handling"""
    error: Exception
    timestamp: datetime
    function_name: str
    args: tuple
    kwargs: dict
    traceback: str
    retry_count: int = 0
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(seconds=60)
    expected_exception: type = Exception
    half_open_requests: int = 3


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self.half_open_requests = 0
        self.success_count = 0
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time >= self.config.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.half_open_requests:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker closed after successful recovery")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened after failure in half-open state")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RetryStrategy:
    """Retry strategy with exponential backoff"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, retry_count: int) -> float:
        """Calculate delay for retry attempt"""
        delay = min(
            self.initial_delay * (self.exponential_base ** retry_count),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random())
        
        return delay


class ErrorRecoveryService:
    """Main error recovery service"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorContext] = []
        self.fallback_handlers: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.max_history_size = 1000
    
    def with_recovery(
        self,
        fallback: Optional[Callable] = None,
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ):
        """Decorator for adding error recovery to functions"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_recovery_async(
                    func, args, kwargs, fallback, circuit_breaker, retry_strategy, severity
                )
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._execute_with_recovery_sync(
                    func, args, kwargs, fallback, circuit_breaker, retry_strategy, severity
                )
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _execute_with_recovery_async(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        fallback: Optional[Callable],
        circuit_breaker_config: Optional[CircuitBreakerConfig],
        retry_strategy: Optional[RetryStrategy],
        severity: ErrorSeverity
    ) -> Any:
        """Execute async function with recovery mechanisms"""
        func_name = func.__name__
        retry_count = 0
        
        # Get or create circuit breaker
        if circuit_breaker_config and func_name not in self.circuit_breakers:
            self.circuit_breakers[func_name] = CircuitBreaker(circuit_breaker_config)
        
        while True:
            try:
                # Try circuit breaker first
                if circuit_breaker_config:
                    return await self.circuit_breakers[func_name].call_async(
                        func, *args, **kwargs
                    )
                else:
                    return await func(*args, **kwargs)
                    
            except Exception as e:
                # Log error context
                error_context = ErrorContext(
                    error=e,
                    timestamp=datetime.now(),
                    function_name=func_name,
                    args=args,
                    kwargs=kwargs,
                    traceback=traceback.format_exc(),
                    retry_count=retry_count,
                    severity=severity
                )
                self._record_error(error_context)
                
                # Check if we should retry
                if retry_strategy and retry_count < retry_strategy.max_retries:
                    retry_count += 1
                    delay = retry_strategy.get_delay(retry_count)
                    logger.warning(
                        f"Retrying {func_name} after {delay:.2f}s "
                        f"(attempt {retry_count}/{retry_strategy.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                # Try fallback
                if fallback:
                    logger.warning(f"Using fallback for {func_name}")
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    else:
                        return fallback(*args, **kwargs)
                
                # No more recovery options
                raise
    
    def _execute_with_recovery_sync(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        fallback: Optional[Callable],
        circuit_breaker_config: Optional[CircuitBreakerConfig],
        retry_strategy: Optional[RetryStrategy],
        severity: ErrorSeverity
    ) -> Any:
        """Execute sync function with recovery mechanisms"""
        func_name = func.__name__
        retry_count = 0
        
        # Get or create circuit breaker
        if circuit_breaker_config and func_name not in self.circuit_breakers:
            self.circuit_breakers[func_name] = CircuitBreaker(circuit_breaker_config)
        
        while True:
            try:
                # Try circuit breaker first
                if circuit_breaker_config:
                    return self.circuit_breakers[func_name].call(
                        func, *args, **kwargs
                    )
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                # Log error context
                error_context = ErrorContext(
                    error=e,
                    timestamp=datetime.now(),
                    function_name=func_name,
                    args=args,
                    kwargs=kwargs,
                    traceback=traceback.format_exc(),
                    retry_count=retry_count,
                    severity=severity
                )
                self._record_error(error_context)
                
                # Check if we should retry
                if retry_strategy and retry_count < retry_strategy.max_retries:
                    retry_count += 1
                    delay = retry_strategy.get_delay(retry_count)
                    logger.warning(
                        f"Retrying {func_name} after {delay:.2f}s "
                        f"(attempt {retry_count}/{retry_strategy.max_retries})"
                    )
                    import time
                    time.sleep(delay)
                    continue
                
                # Try fallback
                if fallback:
                    logger.warning(f"Using fallback for {func_name}")
                    return fallback(*args, **kwargs)
                
                # No more recovery options
                raise
    
    def _record_error(self, error_context: ErrorContext):
        """Record error in history"""
        self.error_history.append(error_context)
        
        # Limit history size
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
        
        # Log based on severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error in {error_context.function_name}: {error_context.error}")
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error in {error_context.function_name}: {error_context.error}")
        else:
            logger.warning(f"Error in {error_context.function_name}: {error_context.error}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {
                "total_errors": 0,
                "errors_by_function": {},
                "errors_by_severity": {},
                "recent_errors": []
            }
        
        errors_by_function = {}
        errors_by_severity = {severity.value: 0 for severity in ErrorSeverity}
        
        for error in self.error_history:
            # Count by function
            if error.function_name not in errors_by_function:
                errors_by_function[error.function_name] = 0
            errors_by_function[error.function_name] += 1
            
            # Count by severity
            errors_by_severity[error.severity.value] += 1
        
        return {
            "total_errors": len(self.error_history),
            "errors_by_function": errors_by_function,
            "errors_by_severity": errors_by_severity,
            "recent_errors": [
                {
                    "function": e.function_name,
                    "error": str(e.error),
                    "timestamp": e.timestamp.isoformat(),
                    "severity": e.severity.value
                }
                for e in self.error_history[-10:]
            ],
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    def reset_circuit_breaker(self, function_name: str):
        """Manually reset a circuit breaker"""
        if function_name in self.circuit_breakers:
            cb = self.circuit_breakers[function_name]
            cb.state = CircuitState.CLOSED
            cb.failure_count = 0
            cb.success_count = 0
            logger.info(f"Circuit breaker for {function_name} manually reset")


# Global instance
error_recovery = ErrorRecoveryService()


# Convenience decorators
def with_retry(max_retries: int = 3, initial_delay: float = 1.0):
    """Decorator for simple retry logic"""
    return error_recovery.with_recovery(
        retry_strategy=RetryStrategy(max_retries=max_retries, initial_delay=initial_delay)
    )


def with_fallback(fallback_func: Callable):
    """Decorator for fallback logic"""
    return error_recovery.with_recovery(fallback=fallback_func)


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception
):
    """Decorator for circuit breaker"""
    return error_recovery.with_recovery(
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=timedelta(seconds=recovery_timeout),
            expected_exception=expected_exception
        )
    ) 