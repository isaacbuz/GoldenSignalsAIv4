from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class Prediction:
    """Model prediction for a market data point."""
    
    timestamp: datetime
    symbol: str
    predicted_price: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate prediction parameters."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Prediction confidence must be between 0 and 1")
            
    @property
    def is_high_confidence(self) -> bool:
        """Check if prediction has high confidence."""
        return self.confidence >= 0.8
        
    @property
    def is_medium_confidence(self) -> bool:
        """Check if prediction has medium confidence."""
        return 0.5 <= self.confidence < 0.8
        
    @property
    def is_low_confidence(self) -> bool:
        """Check if prediction has low confidence."""
        return self.confidence < 0.5
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'predicted_price': self.predicted_price,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Prediction':
        """Create prediction from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=data['symbol'],
            predicted_price=float(data['predicted_price']),
            confidence=float(data['confidence']),
            metadata=data.get('metadata')
        ) 