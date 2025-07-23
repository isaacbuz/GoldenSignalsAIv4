"""Mock market data models for testing"""
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


class MarketData:
    def __init__(self, 
                 symbol: str,
                 data: Optional[pd.DataFrame] = None,
                 timeframe: str = "1h",
                 indicators: Optional[Dict[str, float]] = None,
                 current_price: Optional[float] = None):
        self.symbol = symbol
        self.data = data if data is not None else pd.DataFrame()
        self.timeframe = timeframe
        self.indicators = indicators or {}
        self.timestamp = datetime.now()
        self.current_price = current_price or 100.0
