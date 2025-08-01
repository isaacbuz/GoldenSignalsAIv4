"""
Pytest configuration for agent tests
"""

import pytest
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Simple test-specific fixtures

@pytest.fixture
def sample_signal():
    """Create a sample signal for testing"""
    return {
        'symbol': 'AAPL',
        'action': 'buy',
        'confidence': 0.75,
        'timestamp': '2024-01-01T00:00:00',
        'metadata': {
            'source': 'technical',
            'indicators': ['rsi', 'macd']
        }
    }

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    return {
        'symbol': 'AAPL',
        'price': 150.00,
        'volume': 1000000,
        'high': 152.00,
        'low': 149.00,
        'open': 150.50,
        'close': 151.00
    }
