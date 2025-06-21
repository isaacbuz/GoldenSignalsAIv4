from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class MarketData:
    """Market data point containing price and volume information."""
    
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    trades: int
    interval_seconds: int
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize additional attributes."""
        self.symbol_hash = hash(self.symbol) % 1000  # Create a numeric hash for the symbol
        
    @property
    def price_change(self) -> float:
        """Calculate the price change."""
        return self.close - self.open
        
    @property
    def price_change_pct(self) -> float:
        """Calculate the percentage price change."""
        return (self.price_change / self.open) * 100 if self.open != 0 else 0
        
    @property
    def high_low_range(self) -> float:
        """Calculate the high-low range."""
        return self.high - self.low
        
    @property
    def high_low_range_pct(self) -> float:
        """Calculate the percentage high-low range."""
        return (self.high_low_range / self.low) * 100 if self.low != 0 else 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert market data to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap,
            'trades': self.trades,
            'interval_seconds': self.interval_seconds,
            'metadata': self.metadata or {}
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create market data from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=data['symbol'],
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=float(data['volume']),
            vwap=float(data['vwap']),
            trades=int(data['trades']),
            interval_seconds=int(data['interval_seconds']),
            metadata=data.get('metadata')
        ) 