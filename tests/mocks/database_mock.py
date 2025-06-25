"""Mock database manager for testing."""

from typing import List, Optional, Dict, Any
from datetime import datetime

class MockDatabaseManager:
    def __init__(self):
        self.signals = []
        self.market_data = []
        self.portfolios = {}
        self.users = {}
    
    async def store_signal(self, signal):
        self.signals.append(signal)
        return signal.signal_id if hasattr(signal, 'signal_id') else len(self.signals)
    
    async def get_signals(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        signals = self.signals
        if symbol:
            signals = [s for s in signals if s.get('symbol') == symbol]
        return signals[:limit]
    
    async def get_market_data(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        return self.market_data[:limit]
    
    async def update_agent_performance(self, agent_id: str, metrics: Dict[str, Any]):
        pass
    
    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]):
        pass
    
    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return None
