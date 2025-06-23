"""
Test utilities and helpers
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

def create_test_dataframe(
    symbol: str = "TEST",
    periods: int = 100,
    freq: str = "1h",
    trend: str = "neutral"
) -> pd.DataFrame:
    """Create test market data with specified trend"""
    
    dates = pd.date_range(
        end=datetime.now(timezone.utc),
        periods=periods,
        freq=freq,
        tz=timezone.utc
    )
    
    # Base price
    base_price = 100.0
    
    # Generate price based on trend
    if trend == "bullish":
        prices = base_price + np.cumsum(np.random.uniform(0, 1, periods))
    elif trend == "bearish":
        prices = base_price - np.cumsum(np.random.uniform(0, 1, periods))
    else:  # neutral
        prices = base_price + np.cumsum(np.random.uniform(-0.5, 0.5, periods))
    
    # Add noise
    prices = prices + np.random.normal(0, 0.5, periods)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.uniform(-1, 1, periods),
        'high': prices + np.random.uniform(0, 2, periods),
        'low': prices - np.random.uniform(0, 2, periods),
        'close': prices,
        'volume': np.random.randint(1000, 100000, periods)
    })
    
    # Fix high/low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df

def assert_valid_signal(signal: Dict[str, Any]) -> None:
    """Assert that a signal has valid structure"""
    assert 'id' in signal
    assert 'symbol' in signal
    assert 'action' in signal
    assert signal['action'] in ['BUY', 'SELL', 'HOLD']
    assert 'confidence' in signal
    assert 0 <= signal['confidence'] <= 1
    assert 'price' in signal
    assert signal['price'] > 0
    assert 'timestamp' in signal
