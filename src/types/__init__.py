"""
Type definitions and protocols for GoldenSignalsAI
"""

from typing import TypedDict, Protocol, Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd


class MarketData(TypedDict):
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    

class Signal(TypedDict):
    """Trading signal structure"""
    id: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    timestamp: str
    reason: str
    indicators: Dict[str, float]
    risk_level: str
    entry_price: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any]
    quality_score: float


class RiskAnalysis(TypedDict):
    """Risk analysis result structure"""
    total_risk: float
    position_size: float
    max_loss: float
    risk_reward_ratio: float
    recommendations: List[str]


class DataValidationReport(TypedDict):
    """Data validation report structure"""
    symbol: str
    is_valid: bool
    issues: List[str]
    score: float
    source: Optional[str]


class DataProvider(Protocol):
    """Protocol for data providers"""
    
    async def fetch_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch market data"""
        ...
        
    async def validate_connection(self) -> bool:
        """Validate provider connection"""
        ...


class SignalGenerator(Protocol):
    """Protocol for signal generators"""
    
    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """Generate trading signals"""
        ...
        
    def get_confidence_threshold(self) -> float:
        """Get confidence threshold"""
        ... 