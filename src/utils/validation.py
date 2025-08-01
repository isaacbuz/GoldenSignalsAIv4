"""
Validation utilities for GoldenSignalsAI
"""

import re
from typing import Union


def validate_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format

    Args:
        symbol: Stock symbol to validate

    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False

    # Check length (most symbols are 1-5 characters, some ETFs can be longer)
    if len(symbol) > 6 or len(symbol) < 1:
        return False

    # Allow letters, dots (for BRK.B style), and hyphens (for some special symbols)
    pattern = r"^[A-Z][A-Z0-9]*(\.[A-Z])?$"
    return bool(re.match(pattern, symbol.upper()))


def validate_price(price: Union[float, int, None]) -> bool:
    """
    Validate price value

    Args:
        price: Price to validate

    Returns:
        True if valid, False otherwise
    """
    if price is None:
        return False

    # Check if it's a number
    if not isinstance(price, (int, float)):
        return False

    # Price must be positive
    return price > 0


def validate_timeframe(timeframe: str) -> bool:
    """
    Validate timeframe format

    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')

    Returns:
        True if valid, False otherwise
    """
    if not timeframe or not isinstance(timeframe, str):
        return False

    # Valid timeframes
    valid_timeframes = [
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",  # Minutes
        "1h",
        "2h",
        "4h",  # Hours
        "1d",
        "1w",
        "1M",  # Day, week, month
    ]

    return timeframe in valid_timeframes
