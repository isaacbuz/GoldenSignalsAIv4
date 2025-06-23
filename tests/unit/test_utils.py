"""
Unit tests for utility modules
"""
import pytest
from datetime import datetime, timezone, timedelta
import pytz
from src.utils.metrics import calculate_win_rate, calculate_profit_factor, calculate_sharpe_ratio
from src.utils.timezone_utils import (
    EASTERN_TZ, now_utc, now_eastern, to_eastern, 
    is_market_hours, is_extended_hours, make_aware
)


class TestMetrics:
    """Test cases for metrics utilities"""
    
    def test_calculate_win_rate(self):
        """Test win rate calculation"""
        # All wins
        assert calculate_win_rate(10, 0) == 1.0
        
        # All losses
        assert calculate_win_rate(0, 10) == 0.0
        
        # Mixed
        assert calculate_win_rate(6, 4) == 0.6
        
        # No trades
        assert calculate_win_rate(0, 0) == 0.0
    
    def test_calculate_profit_factor(self):
        """Test profit factor calculation"""
        # Perfect profit factor
        assert calculate_profit_factor([10, 20, 30], []) == float('inf')
        
        # No profits
        assert calculate_profit_factor([], [10, 20]) == 0.0
        
        # Mixed scenario
        assert calculate_profit_factor([100, 50], [25, 25]) == 3.0
        
        # No trades
        assert calculate_profit_factor([], []) == 0.0
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        # Positive returns
        returns = [0.05, 0.03, 0.07, 0.02, 0.06]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0
        
        # Negative returns
        returns = [-0.05, -0.03, -0.02, -0.04]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0
        
        # No returns
        assert calculate_sharpe_ratio([]) == 0.0
        
        # Single return
        assert calculate_sharpe_ratio([0.05]) == 0.0
        
        # Zero volatility (returns inf)
        sharpe_zero_vol = calculate_sharpe_ratio([0.05, 0.05, 0.05])
        assert sharpe_zero_vol == float('inf')


class TestTimezoneUtils:
    """Test cases for timezone utilities"""
    
    def test_now_functions(self):
        """Test current time functions"""
        utc_time = now_utc()
        assert utc_time.tzinfo == timezone.utc
        
        eastern_time = now_eastern()
        assert eastern_time.tzinfo is not None
    
    def test_is_market_hours(self):
        """Test market hours check"""
        # Create a known market open time (Monday 10 AM EST)
        open_time = datetime(2024, 1, 8, 10, 0)  # Monday 10 AM
        open_time = make_aware(open_time, EASTERN_TZ)
        
        # Should be open
        assert is_market_hours(open_time) == True
        
        # Create a known market closed time (Saturday)
        closed_time = datetime(2024, 1, 6, 10, 0)  # Saturday
        closed_time = make_aware(closed_time, EASTERN_TZ)
        
        # Should be closed
        assert is_market_hours(closed_time) == False
        
        # After hours
        after_hours = datetime(2024, 1, 8, 17, 0)  # Monday 5 PM
        after_hours = make_aware(after_hours, EASTERN_TZ)
        assert is_market_hours(after_hours) == False
    
    def test_timezone_conversion(self):
        """Test timezone conversion"""
        # UTC time
        utc_time = datetime(2024, 1, 8, 15, 0, tzinfo=timezone.utc)
        
        # Convert to Eastern
        eastern_time = to_eastern(utc_time)
        
        # Should be 10 AM EST (5 hours behind UTC)
        assert eastern_time.hour == 10
    
    def test_is_extended_hours(self):
        """Test extended hours detection"""
        # Pre-market (Monday 5 AM EST)
        pre_market = datetime(2024, 1, 8, 5, 0)
        pre_market = make_aware(pre_market, EASTERN_TZ)
        assert is_extended_hours(pre_market) == True
        
        # After-hours (Monday 5 PM EST)
        after_hours = datetime(2024, 1, 8, 17, 0)
        after_hours = make_aware(after_hours, EASTERN_TZ)
        assert is_extended_hours(after_hours) == True
        
        # Regular hours (Monday 10 AM EST)
        regular = datetime(2024, 1, 8, 10, 0)
        regular = make_aware(regular, EASTERN_TZ)
        assert is_extended_hours(regular) == False
    
    def test_make_aware(self):
        """Test making datetime timezone aware"""
        # Naive datetime
        naive_dt = datetime(2024, 1, 8, 10, 0)
        aware_dt = make_aware(naive_dt)
        
        assert aware_dt.tzinfo is not None
        assert aware_dt.tzinfo == timezone.utc
    
    def test_edge_cases(self):
        """Test edge cases for timezone utils"""
        # Weekend check
        saturday = datetime(2024, 1, 6, 10, 0)  # Saturday
        saturday = make_aware(saturday, EASTERN_TZ)
        assert is_market_hours(saturday) == False
        assert is_extended_hours(saturday) == False


class TestErrorRecovery:
    """Test cases for error recovery utilities"""
    
    def test_retry_decorator(self):
        """Test retry decorator functionality"""
        from src.utils.error_recovery import retry
        
        call_count = 0
        
        @retry(max_attempts=3, delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
    
    def test_exponential_backoff(self):
        """Test exponential backoff"""
        from src.utils.error_recovery import exponential_backoff
        
        delays = []
        for attempt in range(5):
            delay = exponential_backoff(attempt)
            delays.append(delay)
        
        # Delays should increase exponentially
        for i in range(1, len(delays)):
            assert delays[i] > delays[i-1]
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        from src.utils.error_recovery import CircuitBreaker, CircuitBreakerConfig, CircuitState
        
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=timedelta(seconds=1))
        breaker = CircuitBreaker(config)
        
        # Should start closed
        assert breaker.state == CircuitState.CLOSED
        
        # Test function that fails
        def failing_func():
            raise Exception("Test failure")
        
        # Record failures
        for _ in range(3):
            try:
                breaker.call(failing_func)
            except:
                pass
        
        # Should be open after threshold
        assert breaker.state == CircuitState.OPEN
    
    def test_fallback_handler(self):
        """Test fallback handler"""
        from src.utils.error_recovery import with_fallback
        
        @with_fallback(lambda: "fallback_value")
        def risky_function():
            raise Exception("Something went wrong")
        
        result = risky_function()
        assert result == "fallback_value"


class TestValidation:
    """Test cases for validation utilities"""
    
    def test_validate_symbol(self):
        """Test symbol validation"""
        from src.utils.validation import validate_symbol
        
        # Valid symbols
        assert validate_symbol("AAPL") == True
        assert validate_symbol("GOOGL") == True
        assert validate_symbol("BRK.B") == True
        
        # Invalid symbols
        assert validate_symbol("") == False
        assert validate_symbol("TOOLONGSYMBOL") == False
        assert validate_symbol("123") == False
        assert validate_symbol("@#$") == False
    
    def test_validate_price(self):
        """Test price validation"""
        from src.utils.validation import validate_price
        
        # Valid prices
        assert validate_price(100.50) == True
        assert validate_price(0.01) == True
        assert validate_price(10000) == True
        
        # Invalid prices
        assert validate_price(-10) == False
        assert validate_price(0) == False
        assert validate_price(None) == False
        assert validate_price("100") == False
    
    def test_validate_timeframe(self):
        """Test timeframe validation"""
        from src.utils.validation import validate_timeframe
        
        # Valid timeframes
        assert validate_timeframe("1m") == True
        assert validate_timeframe("5m") == True
        assert validate_timeframe("1h") == True
        assert validate_timeframe("1d") == True
        
        # Invalid timeframes
        assert validate_timeframe("2m") == False
        assert validate_timeframe("invalid") == False
        assert validate_timeframe("") == False 