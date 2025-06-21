"""
Unit tests for timezone utilities
"""

import pytest
from datetime import datetime, timezone
import pytz

from src.utils.timezone_utils import (
    now_utc,
    make_aware,
    to_utc,
    is_market_hours,
    to_eastern,
    format_datetime,
    parse_datetime,
    is_extended_hours,
    get_market_calendar
)


@pytest.mark.unit
class TestTimezoneUtils:
    """Test cases for timezone utilities"""
    
    def test_now_utc(self):
        """Test UTC now function"""
        result = now_utc()
        assert result.tzinfo == timezone.utc
        assert isinstance(result, datetime)
    
    def test_make_aware(self):
        """Test make_aware function"""
        # Test with naive datetime
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        result = make_aware(naive_dt)
        assert result.tzinfo == timezone.utc
        assert result.hour == 12
        
        # Test with UTC datetime
        utc_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = make_aware(utc_dt)
        assert result == utc_dt
        
        # Test with other timezone
        eastern = pytz.timezone('US/Eastern')
        eastern_dt = eastern.localize(datetime(2024, 1, 1, 12, 0, 0))
        result = make_aware(eastern_dt)
        assert result.tzinfo == timezone.utc
        assert result.hour == 17  # 12 PM EST = 5 PM UTC
    
    def test_to_utc(self):
        """Test to UTC conversion"""
        # Test datetime conversion
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = to_utc(dt)
        assert result.tzinfo == timezone.utc
        
        # Test with timezone aware datetime
        eastern = pytz.timezone('US/Eastern')
        eastern_dt = eastern.localize(datetime(2024, 1, 1, 12, 0, 0))
        result = to_utc(eastern_dt)
        assert result.tzinfo == timezone.utc
        assert result.hour == 17  # 12 PM EST = 5 PM UTC
    
    def test_is_market_hours(self):
        """Test market hours check"""
        # During market hours (Monday 10 AM EST)
        market_open = datetime(2024, 1, 8, 15, 0, 0, tzinfo=timezone.utc)  # Monday 10 AM EST
        assert is_market_hours(market_open) is True
        
        # After market hours (Monday 5 PM EST)
        market_closed = datetime(2024, 1, 8, 22, 0, 0, tzinfo=timezone.utc)  # Monday 5 PM EST
        assert is_market_hours(market_closed) is False
        
        # Weekend
        weekend = datetime(2024, 1, 6, 15, 0, 0, tzinfo=timezone.utc)  # Saturday
        assert is_market_hours(weekend) is False
    
    def test_to_eastern(self):
        """Test to_eastern conversion"""
        utc_dt = datetime(2024, 1, 1, 17, 0, 0, tzinfo=timezone.utc)
        
        # Convert to Eastern
        eastern_dt = to_eastern(utc_dt)
        assert eastern_dt.hour == 12  # 5 PM UTC = 12 PM EST
        
        # Test with naive datetime
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        eastern_dt = to_eastern(naive_dt)
        assert eastern_dt.hour == 7  # 12 PM UTC = 7 AM EST
    
    def test_format_datetime(self):
        """Test datetime formatting"""
        dt = datetime(2024, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
        
        # Default format
        result = format_datetime(dt)
        assert "2024-01-01 12:30:45" in result
        
        # Custom format
        result = format_datetime(dt, "%Y-%m-%d %H:%M")
        assert result == "2024-01-01 12:30"
    
    def test_parse_datetime(self):
        """Test datetime parsing"""
        # ISO format
        result = parse_datetime("2024-01-01T12:30:45+00:00")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12
        assert result.minute == 30
        assert result.second == 45
        assert result.tzinfo == timezone.utc
        
        # Without timezone (assumes UTC)
        result = parse_datetime("2024-01-01T12:30:45")
        assert result.tzinfo == timezone.utc
    
    def test_is_extended_hours(self):
        """Test extended hours check"""
        # Pre-market hours (6 AM EST)
        pre_market = datetime(2024, 1, 8, 11, 0, 0, tzinfo=timezone.utc)  # Monday 6 AM EST
        assert is_extended_hours(pre_market) is True
        
        # After-hours (5 PM EST)
        after_hours = datetime(2024, 1, 8, 22, 0, 0, tzinfo=timezone.utc)  # Monday 5 PM EST
        assert is_extended_hours(after_hours) is True
        
        # Regular hours
        regular = datetime(2024, 1, 8, 15, 0, 0, tzinfo=timezone.utc)  # Monday 10 AM EST
        assert is_extended_hours(regular) is False
        
        # Weekend
        weekend = datetime(2024, 1, 6, 11, 0, 0, tzinfo=timezone.utc)  # Saturday
        assert is_extended_hours(weekend) is False 