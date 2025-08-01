"""Signal domain model."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Signal types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(str, Enum):
    """Signal strength levels."""

    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"


class Signal(BaseModel):
    """Trading signal domain model."""

    id: Optional[str] = None
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    reasoning: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    indicators: Optional[Dict[str, float]] = None
    current_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_score: Optional[float] = None
    market_conditions: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}
