"""
Unit tests for Error Recovery System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import time
from unittest.mock import Mock, patch

from src.utils.error_recovery import (
    ErrorRecoveryService,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryStrategy,
    ErrorSeverity,
    with_retry,
    with_fallback,
    with_circuit_breaker,
    error_recovery
)


@pytest.mark.unit
class TestRetryStrategy:
    """Test cases for RetryStrategy"""
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation"""
        strategy = RetryStrategy(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
        
        assert strategy.get_delay(0) == 1.0
        assert strategy.get_delay(1) == 2.0
        assert strategy.get_delay(2) == 4.0
        assert strategy.get_delay(3) == 8.0
    
    def test_max_delay(self):
        """Test max delay limit"""
        strategy = RetryStrategy(
            initial_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=False
        )
        
        assert strategy.get_delay(0) == 1.0
        assert strategy.get_delay(2) == 4.0
        assert strategy.get_delay(3) == 5.0  # Capped at max_delay
        assert strategy.get_delay(10) == 5.0  # Still capped
    
    def test_jitter(self):
        """Test jitter adds randomness"""
        strategy = RetryStrategy(
            initial_delay=1.0,
            jitter=True
        )
        
        delays = [strategy.get_delay(1) for _ in range(10)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1
        # But should be within expected range
        assert all(1.0 <= d <= 4.0 for d in delays)


@pytest.mark.unit
class TestCircuitBreaker:
    """Test cases for CircuitBreaker"""
    
    def test_initial_state(self):
        """Test circuit breaker starts closed"""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker(config)
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_success_keeps_closed(self):
        """Test successful calls keep circuit closed"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)
        
        def success_func():
            return "success"
        
        for _ in range(5):
            result = cb.call(success_func)
            assert result == "success"
            assert cb.state == CircuitState.CLOSED
    
    def test_failures_open_circuit(self):
        """Test failures open the circuit"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)
        
        def failing_func():
            raise Exception("Test failure")
        
        # First failures
        for i in range(2):
            with pytest.raises(Exception):
                cb.call(failing_func)
            assert cb.state == CircuitState.CLOSED
        
        # Third failure opens circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
    
    def test_open_circuit_rejects_calls(self):
        """Test open circuit rejects calls immediately"""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)
        
        # Open the circuit
        with pytest.raises(Exception):
            cb.call(lambda: 1/0)
        
        assert cb.state == CircuitState.OPEN
        
        # Next call should be rejected
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(lambda: "success")
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test circuit breaker with async functions"""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config)
        
        async def async_failing():
            raise ValueError("Async failure")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call_async(async_failing)
        
        assert cb.state == CircuitState.OPEN


@pytest.mark.unit
class TestErrorRecoveryService:
    """Test cases for ErrorRecoveryService"""
    
    @pytest.fixture
    def recovery_service(self):
        """Create a fresh recovery service"""
        return ErrorRecoveryService()
    
    def test_retry_decorator(self):
        """Test retry decorator"""
        call_count = 0
        
        @with_retry(max_retries=3, initial_delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhaustion(self):
        """Test retry exhaustion raises exception"""
        @with_retry(max_retries=2, initial_delay=0.01)
        def always_fails():
            raise ValueError("Permanent failure")
        
        with pytest.raises(ValueError, match="Permanent failure"):
            always_fails()
    
    def test_fallback_decorator(self):
        """Test fallback decorator"""
        def fallback_func():
            return "fallback result"
        
        @with_fallback(fallback_func)
        def failing_function():
            raise Exception("Primary failure")
        
        result = failing_function()
        assert result == "fallback result"
    
    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test retry with async function"""
        call_count = 0
        
        @with_retry(max_retries=2, initial_delay=0.01)
        async def async_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Async temporary failure")
            return "async success"
        
        result = await async_flaky()
        assert result == "async success"
        assert call_count == 2
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator"""
        call_count = 0
        
        @with_circuit_breaker(failure_threshold=2, recovery_timeout=1)
        def protected_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Service unavailable")
            return "recovered"
        
        # First two calls fail and open circuit
        for _ in range(2):
            with pytest.raises(Exception, match="Service unavailable"):
                protected_function()
        
        # Circuit should be open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            protected_function()
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Circuit should attempt recovery
        result = protected_function()
        assert result == "recovered"
    
    def test_error_statistics(self, recovery_service):
        """Test error statistics tracking"""
        @recovery_service.with_recovery(severity=ErrorSeverity.HIGH)
        def error_function():
            raise ValueError("Test error")
        
        # Generate some errors
        for _ in range(3):
            try:
                error_function()
            except ValueError:
                pass
        
        stats = recovery_service.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert "error_function" in stats["errors_by_function"]
        assert stats["errors_by_function"]["error_function"] == 3
        assert stats["errors_by_severity"]["high"] == 3
        assert len(stats["recent_errors"]) == 3
    
    def test_combined_recovery(self):
        """Test combined retry and fallback"""
        call_count = 0
        
        def fallback():
            return "fallback"
        
        @error_recovery.with_recovery(
            retry_strategy=RetryStrategy(max_retries=2, initial_delay=0.01),
            fallback=fallback
        )
        def complex_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")
        
        result = complex_function()
        assert result == "fallback"
        assert call_count == 3  # Initial + 2 retries
    
    def test_manual_circuit_reset(self, recovery_service):
        """Test manual circuit breaker reset"""
        @recovery_service.with_recovery(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1)
        )
        def protected_func():
            raise Exception("Failure")
        
        # Open the circuit
        with pytest.raises(Exception):
            protected_func()
        
        # Verify circuit is open
        cb = recovery_service.circuit_breakers["protected_func"]
        assert cb.state == CircuitState.OPEN
        
        # Reset manually
        recovery_service.reset_circuit_breaker("protected_func")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0 