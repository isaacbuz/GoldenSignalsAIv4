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

    async def get_cached_ohlcv_data(
        self, symbol: str, timeframe: str
    ) -> Optional[List[Dict[str, Any]]]:
        return None

    async def cache_ohlcv_data(
        self, symbol: str, timeframe: str, data: List[Dict[str, Any]]
    ) -> None:
        self.cache[f"ohlcv:{symbol}:{timeframe}"] = data

    async def get_cached_latest_signals(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        return self.streams.get(symbol, [])

    async def store_temp_data(self, key: str, value: Any, ttl: int) -> None:
        self.cache[key] = value

    async def get_temp_data(self, key: str) -> Any:
        return self.cache.get(key)
