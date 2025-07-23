"""
Pydantic models for request/response schemas
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SignalCreate(BaseModel):
    """Schema for creating a new signal"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    signal_type: str = Field(..., description="Type of signal (BUY, SELL, HOLD)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence score")
    strength: str = Field(..., description="Signal strength (STRONG, MODERATE, WEAK)")
    source: str = Field(..., description="Signal source/agent name")
    current_price: Optional[float] = Field(None, description="Current price when signal was generated")
    target_price: Optional[float] = Field(None, description="Target price for the signal")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Signal generation timestamp")
    expiry: Optional[datetime] = Field(None, description="Signal expiry timestamp")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional signal metadata")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Risk score")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the signal") 