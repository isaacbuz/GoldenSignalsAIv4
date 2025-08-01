"""
Timezone utilities for consistent datetime handling across the application
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from zoneinfo import ZoneInfo

import pytz

# Default timezone for the application (UTC)
DEFAULT_TIMEZONE = timezone.utc

# Common timezones
EASTERN_TZ = ZoneInfo("America/New_York")
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
LONDON_TZ = ZoneInfo("Europe/London")
TOKYO_TZ = ZoneInfo("Asia/Tokyo")


def now_utc() -> datetime:
    """Get current time in UTC with timezone awareness"""
    return datetime.now(timezone.utc)


def now_eastern() -> datetime:
    """Get current time in Eastern timezone"""
    return datetime.now(EASTERN_TZ)


def make_aware(dt: datetime, tz: Optional[timezone] = None) -> datetime:
    """
    Make a naive datetime timezone-aware

    Args:
        dt: Datetime object (naive or aware)
        tz: Target timezone (defaults to UTC)

    Returns:
        Timezone-aware datetime
    """
    if tz is None:
        tz = DEFAULT_TIMEZONE

    if dt.tzinfo is None:
        # Naive datetime - assume it's in the target timezone
        return dt.replace(tzinfo=tz)
    else:
        # Already aware - convert to target timezone
        return dt.astimezone(tz)


def to_utc(dt: datetime) -> datetime:
    """Convert any datetime to UTC"""
    if dt.tzinfo is None:
        # Assume naive datetime is in local time
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_eastern(dt: datetime) -> datetime:
    """Convert any datetime to Eastern timezone"""
    if dt.tzinfo is None:
        dt = make_aware(dt)
    return dt.astimezone(EASTERN_TZ)


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if given time is during market hours (9:30 AM - 4:00 PM ET)

    Args:
        dt: Datetime to check (defaults to current time)

    Returns:
        True if market is open
    """
    if dt is None:
        dt = now_eastern()
    else:
        dt = to_eastern(dt)

    # Check if it's a weekday
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Check time
    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= dt <= market_close


def is_extended_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if given time is during extended trading hours
    Pre-market: 4:00 AM - 9:30 AM ET
    After-hours: 4:00 PM - 8:00 PM ET
    """
    if dt is None:
        dt = now_eastern()
    else:
        dt = to_eastern(dt)

    # Check if it's a weekday
    if dt.weekday() >= 5:
        return False

    # Pre-market
    pre_market_start = dt.replace(hour=4, minute=0, second=0, microsecond=0)
    pre_market_end = dt.replace(hour=9, minute=30, second=0, microsecond=0)

    # After-hours
    after_hours_start = dt.replace(hour=16, minute=0, second=0, microsecond=0)
    after_hours_end = dt.replace(hour=20, minute=0, second=0, microsecond=0)

    return (pre_market_start <= dt < pre_market_end) or (after_hours_start < dt <= after_hours_end)


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """Format datetime with timezone info"""
    if dt.tzinfo is None:
        dt = make_aware(dt)
    return dt.strftime(fmt)


def parse_datetime(dt_str: str, tz: Optional[timezone] = None) -> datetime:
    """
    Parse datetime string and make it timezone-aware

    Handles multiple formats:
    - ISO format with timezone
    - ISO format without timezone
    - Common date formats
    """
    if tz is None:
        tz = DEFAULT_TIMEZONE

    # Try parsing with timezone info first
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.astimezone(tz)
    except:
        pass

    # Try common formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return make_aware(dt, tz)
        except:
            continue

    raise ValueError(f"Unable to parse datetime string: {dt_str}")


def get_market_calendar(start_date: datetime, end_date: datetime) -> list[datetime]:
    """
    Get list of market open days between two dates

    Note: This is a simple implementation that doesn't account for holidays
    For production, use pandas_market_calendars or similar
    """
    current = to_eastern(start_date).date()
    end = to_eastern(end_date).date()
    market_days = []

    while current <= end:
        if current.weekday() < 5:  # Monday-Friday
            market_days.append(datetime.combine(current, datetime.min.time(), tzinfo=EASTERN_TZ))
        current += timedelta(days=1)

    return market_days


# Utility functions for backward compatibility
def utc_now() -> datetime:
    """Alias for now_utc()"""
    return now_utc()


def ensure_timezone_aware(dt: Union[datetime, str, None]) -> Optional[datetime]:
    """
    Ensure datetime is timezone-aware, handling various input types

    Args:
        dt: Datetime object, string, or None

    Returns:
        Timezone-aware datetime or None
    """
    if dt is None:
        return None

    if isinstance(dt, str):
        return parse_datetime(dt)

    if isinstance(dt, datetime):
        return make_aware(dt)

    raise TypeError(f"Unsupported type for datetime conversion: {type(dt)}")
