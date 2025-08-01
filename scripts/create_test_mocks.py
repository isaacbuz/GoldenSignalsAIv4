#!/usr/bin/env python3
"""
Create comprehensive mock modules for testing
"""

import os
import sys

def create_mock_module(module_path: str, content: str):
    """Create a mock module with the given content"""
    # Create directory structure
    dir_path = os.path.dirname(module_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Write the mock module
    with open(module_path, 'w') as f:
        f.write(content)

    print(f"Created mock: {module_path}")

def create_test_mocks():
    """Create all necessary mock modules"""

    # Mock database manager
    create_mock_module('src/core/database.py', '''
"""Mock database module for testing"""
from typing import Any, Dict, List, Optional
from datetime import datetime

class DatabaseManager:
    """Mock database manager"""
    def __init__(self, *args, **kwargs):
        self.signals = []
        self.agent_states = {}
        self.performance_data = {}

    async def store_signal(self, signal_data: Dict[str, Any]) -> None:
        self.signals.append(signal_data)

    async def update_agent_performance(self, agent_id: str, data: Dict[str, Any]) -> None:
        self.performance_data[agent_id] = data

    async def save_agent_state(self, agent_id: str, name: str, state: Dict[str, Any]) -> None:
        self.agent_states[agent_id] = state

    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self.agent_states.get(agent_id)

    async def get_market_data(self, symbol: str, since: datetime, limit: int) -> List[Any]:
        return []

    async def get_signals(self, **kwargs) -> List[Any]:
        return []
''')

    # Mock Redis manager
    create_mock_module('src/core/redis_manager.py', '''
"""Mock Redis module for testing"""
from typing import Any, Dict, List, Optional

class RedisManager:
    """Mock Redis manager"""
    def __init__(self, *args, **kwargs):
        self.cache = {}
        self.streams = {}

    async def add_signal_to_stream(self, symbol: str, signal_data: Dict[str, Any]) -> None:
        if symbol not in self.streams:
            self.streams[symbol] = []
        self.streams[symbol].append(signal_data)

    async def cache_agent_performance(self, agent_id: str, data: Dict[str, Any]) -> None:
        self.cache[f"perf:{agent_id}"] = data

    async def cache_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        self.cache[f"state:{agent_id}"] = state

    async def get_cached_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(f"state:{agent_id}")

    async def get_cached_ohlcv_data(self, symbol: str, timeframe: str) -> Optional[List[Dict[str, Any]]]:
        return None

    async def cache_ohlcv_data(self, symbol: str, timeframe: str, data: List[Dict[str, Any]]) -> None:
        self.cache[f"ohlcv:{symbol}:{timeframe}"] = data

    async def get_cached_latest_signals(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        return self.streams.get(symbol, [])

    async def store_temp_data(self, key: str, value: Any, ttl: int) -> None:
        self.cache[key] = value

    async def get_temp_data(self, key: str) -> Any:
        return self.cache.get(key)
''')

    # Mock ML models
    create_mock_module('src/ml/models/signals.py', '''
"""Mock signal models for testing"""
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class SignalStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

class SignalSource(Enum):
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    FLOW_ANALYSIS = "flow_analysis"
    OPTIONS_ANALYSIS = "options_analysis"
    RISK_ANALYSIS = "risk_analysis"

class Signal:
    def __init__(self,
                 symbol: str,
                 signal_type: SignalType,
                 confidence: float,
                 strength: SignalStrength = SignalStrength.MODERATE,
                 source: SignalSource = SignalSource.TECHNICAL_ANALYSIS,
                 current_price: float = 100.0,
                 target_price: Optional[float] = None,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None,
                 risk_score: float = 0.5,
                 reasoning: str = "",
                 features: Optional[Dict[str, Any]] = None,
                 indicators: Optional[Dict[str, float]] = None,
                 market_conditions: Optional[Dict[str, Any]] = None):
        self.signal_id = str(uuid.uuid4())
        self.symbol = symbol
        self.signal_type = signal_type
        self.confidence = confidence
        self.strength = strength
        self.source = source
        self.current_price = current_price
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_score = risk_score
        self.reasoning = reasoning
        self.features = features or {}
        self.indicators = indicators or {}
        self.market_conditions = market_conditions or {}
        self.created_at = datetime.now()
''')

    # Mock market data
    create_mock_module('src/ml/models/market_data.py', '''
"""Mock market data models for testing"""
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

class MarketData:
    def __init__(self,
                 symbol: str,
                 data: Optional[pd.DataFrame] = None,
                 timeframe: str = "1h",
                 indicators: Optional[Dict[str, float]] = None):
        self.symbol = symbol
        self.data = data if data is not None else pd.DataFrame()
        self.timeframe = timeframe
        self.indicators = indicators or {}
        self.timestamp = datetime.now()
''')

    # Mock metrics
    create_mock_module('src/utils/metrics.py', '''
"""Mock metrics module for testing"""
from typing import Any

class MetricsCollector:
    def __init__(self, name: str):
        self.name = name
        self.metrics = {}

    def increment(self, metric: str) -> None:
        if metric not in self.metrics:
            self.metrics[metric] = 0
        self.metrics[metric] += 1

    def record(self, metric: str, value: float) -> None:
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics[metric].append(value)
''')

    # Mock settings
    create_mock_module('src/config/settings.py', '''
"""Mock settings module for testing"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "GoldenSignalsAI"
    debug: bool = True
    database_url: str = "postgresql://test:test@localhost/test"
    redis_url: str = "redis://localhost:6379"

settings = Settings()
''')

    # Create __init__.py files
    for dir_path in ['src', 'src/core', 'src/ml', 'src/ml/models', 'src/utils', 'src/config']:
        init_file = os.path.join(dir_path, '__init__.py')
        if not os.path.exists(init_file):
            create_mock_module(init_file, '"""Mock module"""')

def main():
    """Main function"""
    print("Creating test mock modules...")
    create_test_mocks()
    print("\n✅ Mock modules created successfully!")

if __name__ == "__main__":
    main()
