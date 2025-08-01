"""
Service interfaces
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class ISignalService(ABC):
    @abstractmethod
    async def generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate trading signal"""
        pass

    @abstractmethod
    async def bulk_generate_signals(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate signals for multiple symbols"""
        pass


class IMarketDataService(ABC):
    @abstractmethod
    async def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data as DataFrame"""
        pass

    @abstractmethod
    async def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate market data"""
        pass
