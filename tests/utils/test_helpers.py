"""Test helper functions."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_market_data(symbol="TEST", days=30):
    """Create sample market data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    data = pd.DataFrame({
        'open': np.random.uniform(90, 110, days),
        'high': np.random.uniform(95, 115, days),
        'low': np.random.uniform(85, 105, days),
        'close': np.random.uniform(90, 110, days),
        'volume': np.random.randint(1000000, 5000000, days)
    }, index=dates)
    return data

def create_mock_response(status_code=200, json_data=None):
    """Create a mock HTTP response."""
    class MockResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self._json_data = json_data or {}
        
        def json(self):
            return self._json_data
    
    return MockResponse(status_code, json_data)
