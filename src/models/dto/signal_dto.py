"""
Signal Data Transfer Objects
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class SignalResponse(BaseModel):
    """Response model for a single signal"""
    id: str
    symbol: str
    action: str
    confidence: float = Field(ge=0, le=1)
    price: float
    timestamp: str
    reason: str
    indicators: Dict[str, float]
    risk_level: str
    entry_price: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    quality_score: float = Field(ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "AAPL_1234567890",
                "symbol": "AAPL",
                "action": "BUY",
                "confidence": 0.75,
                "price": 150.50,
                "timestamp": "2024-01-10T14:30:00Z",
                "reason": "RSI oversold; MACD bullish crossover",
                "indicators": {
                    "rsi": 28.5,
                    "macd": 1.2,
                    "sma_20": 148.3
                },
                "risk_level": "medium",
                "entry_price": 150.50,
                "stop_loss": 147.00,
                "take_profit": 156.00,
                "metadata": {},
                "quality_score": 0.85
            }
        }


class SignalsResponse(BaseModel):
    """Response model for multiple signals"""
    signals: List[Dict[str, Any]]
    count: int
    status: str = "success"
    
    class Config:
        json_schema_extra = {
            "example": {
                "signals": [
                    {
                        "id": "AAPL_1234567890",
                        "symbol": "AAPL",
                        "action": "BUY",
                        "confidence": 0.75,
                        "price": 150.50,
                        "timestamp": "2024-01-10T14:30:00Z",
                        "reason": "RSI oversold; MACD bullish crossover",
                        "indicators": {"rsi": 28.5},
                        "risk_level": "medium",
                        "entry_price": 150.50,
                        "stop_loss": 147.00,
                        "take_profit": 156.00,
                        "metadata": {},
                        "quality_score": 0.85
                    }
                ],
                "count": 1,
                "status": "success"
            }
        }


class SignalCreate(BaseModel):
    """Model for creating a new signal"""
    symbol: str
    action: str = Field(pattern="^(BUY|SELL|HOLD)$")
    confidence: float = Field(ge=0, le=1)
    reason: str
    indicators: Dict[str, float]
    risk_level: str = Field(pattern="^(low|medium|high)$")
    entry_price: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any] = Field(default_factory=dict) 