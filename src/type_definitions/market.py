"""
Market-related type definitions
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Protocol, TypedDict

import pandas as pd

# Type aliases
Symbol = str
Price = Decimal
Volume = int
Timestamp = datetime

# Literal types
SignalAction = Literal["BUY", "SELL", "HOLD"]
Timeframe = Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
OrderType = Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]

# TypedDict definitions
class MarketData(TypedDict):
    symbol: Symbol
    timestamp: Timestamp
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume
    
class TradingSignal(TypedDict):
    id: str
    symbol: Symbol
    action: SignalAction
    confidence: float
    price: Price
    timestamp: Timestamp
    metadata: Dict[str, Any]
    
class Order(TypedDict):
    id: str
    symbol: Symbol
    type: OrderType
    side: SignalAction
    quantity: float
    price: Optional[Price]
    timestamp: Timestamp
    
# Protocol definitions
class MarketDataProvider(Protocol):
    async def get_data(self, symbol: Symbol, timeframe: Timeframe) -> pd.DataFrame:
        ...
        
    async def get_latest_price(self, symbol: Symbol) -> Price:
        ...
        
class SignalGenerator(Protocol):
    async def generate_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        ...
