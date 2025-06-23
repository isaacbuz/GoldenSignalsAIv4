"""
Repository interfaces
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime

class IMarketRepository(ABC):
    @abstractmethod
    async def get_market_data(self, symbol: str, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get market data for symbol"""
        pass
        
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        pass

class ISignalRepository(ABC):
    @abstractmethod
    async def save_signal(self, signal: Dict[str, Any]) -> str:
        """Save trading signal"""
        pass
        
    @abstractmethod
    async def get_signals(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get signals for symbol"""
        pass
