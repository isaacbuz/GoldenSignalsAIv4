"""Mock signal models for testing"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SignalStrength(Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    MODERATE = "moderate"
    STRONG = "strong"


class SignalSource(Enum):
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    FLOW_ANALYSIS = "flow_analysis"
    OPTIONS_ANALYSIS = "options_analysis"
    RISK_ANALYSIS = "risk_analysis"


class Signal:
    def __init__(
        self,
        symbol: str,
        signal_type: SignalType,
        confidence: float,
        strength: SignalStrength = SignalStrength.MODERATE,
        source: SignalSource = SignalSource.TECHNICAL_ANALYSIS,
        current_price: float = 100.0,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_score: float = 0.5,
        reasoning: str = "",
        features: Optional[Dict[str, Any]] = None,
        indicators: Optional[Dict[str, float]] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
    ):
        self.signal_id = str(uuid.uuid4())
        self.symbol = symbol
        self.signal_type = signal_type
        self.confidence = confidence
        self.strength = strength
        self.source = source
        self.current_price = current_price
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_score = risk_score
        self.reasoning = reasoning
        self.features = features or {}
        self.indicators = indicators or {}
        self.market_conditions = market_conditions or {}
        self.created_at = datetime.now()
