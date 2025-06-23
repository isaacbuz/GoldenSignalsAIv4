"""
Signal Models for GoldenSignalsAI V3
"""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class SignalType(str, Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalStrength(str, Enum):
    """Signal strength categories"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalSource(str, Enum):
    """Sources that can generate signals"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    MARKET_REGIME = "market_regime"
    OPTIONS_FLOW = "options_flow"
    MACROECONOMIC = "macroeconomic"
    RISK_MANAGEMENT = "risk_management"
    META_AGENT = "meta_agent"


class Signal(BaseModel):
    """
    Core signal model representing a trading recommendation
    """
    
    # Core identification
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Signal properties
    symbol: str = Field(..., description="Trading symbol (e.g., AAPL, SPY)")
    signal_type: SignalType = Field(..., description="Type of signal")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    strength: SignalStrength = Field(..., description="Signal strength category")
    source: SignalSource = Field(..., description="Agent that generated the signal")
    
    # Price and timing information
    current_price: Optional[float] = Field(None, gt=0, description="Current market price")
    target_price: Optional[float] = Field(None, gt=0, description="Target price")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[float] = Field(None, gt=0, description="Take profit price")
    time_horizon: Optional[int] = Field(None, gt=0, description="Expected time horizon in minutes")
    
    # Risk metrics
    max_drawdown: Optional[float] = Field(None, ge=0.0, le=1.0, description="Maximum acceptable drawdown")
    position_size: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recommended position size")
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Risk assessment score")
    
    # Metadata and context
    reasoning: Optional[str] = Field(None, description="Human-readable reasoning")
    features: Dict[str, Any] = Field(default_factory=dict, description="Feature values used")
    indicators: Dict[str, float] = Field(default_factory=dict, description="Technical indicators")
    market_conditions: Dict[str, Any] = Field(default_factory=dict, description="Market context")
    
    # Validation and quality
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Signal quality score")
    expires_at: Optional[datetime] = Field(None, description="Signal expiration time")
    
    @validator("strength", pre=True, always=True)
    def determine_strength(cls, v, values):
        """Auto-determine strength based on confidence if not provided"""
        if v is None and "confidence" in values:
            confidence = values["confidence"]
            if confidence >= 0.9:
                return SignalStrength.VERY_STRONG
            elif confidence >= 0.75:
                return SignalStrength.STRONG
            elif confidence >= 0.6:
                return SignalStrength.MODERATE
            else:
                return SignalStrength.WEAK
        return v
    
    @validator("expires_at", pre=True, always=True)
    def set_default_expiry(cls, v, values):
        """Set default expiry if not provided"""
        if v is None and "time_horizon" in values and values["time_horizon"]:
            return datetime.utcnow() + timedelta(minutes=values["time_horizon"])
        elif v is None:
            return datetime.utcnow() + timedelta(hours=24)  # Default 24 hour expiry
        return v
    
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        return self.expires_at and datetime.utcnow() > self.expires_at
    
    def get_expected_return(self) -> Optional[float]:
        """Calculate expected return percentage"""
        if self.current_price and self.target_price:
            if self.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                return (self.target_price - self.current_price) / self.current_price
            elif self.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                return (self.current_price - self.target_price) / self.current_price
        return None
    
    def get_risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-reward ratio"""
        if self.current_price and self.target_price and self.stop_loss:
            potential_profit = abs(self.target_price - self.current_price)
            potential_loss = abs(self.current_price - self.stop_loss)
            if potential_loss > 0:
                return potential_profit / potential_loss
        return None


class MetaSignal(BaseModel):
    """
    Meta-signal combining multiple agent signals with consensus
    """
    
    # Core identification
    meta_signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Aggregated signal properties
    symbol: str = Field(..., description="Trading symbol")
    consensus_signal: SignalType = Field(..., description="Consensus signal type")
    consensus_confidence: float = Field(..., ge=0.0, le=1.0, description="Consensus confidence")
    consensus_strength: SignalStrength = Field(..., description="Consensus strength")
    
    # Component signals
    component_signals: List[Signal] = Field(..., description="Individual agent signals")
    agent_weights: Dict[str, float] = Field(..., description="Weight assigned to each agent")
    
    # Consensus metrics
    agreement_score: float = Field(..., ge=0.0, le=1.0, description="How much agents agree")
    uncertainty: float = Field(..., ge=0.0, le=1.0, description="Uncertainty in consensus")
    reliability: float = Field(..., ge=0.0, le=1.0, description="Historical reliability")
    
    # Aggregated financial metrics
    weighted_target_price: Optional[float] = Field(None, description="Weighted target price")
    weighted_stop_loss: Optional[float] = Field(None, description="Weighted stop loss")
    aggregated_risk_score: Optional[float] = Field(None, description="Aggregated risk score")
    
    # Execution recommendations
    recommended_position_size: Optional[float] = Field(None, ge=0.0, le=1.0)
    execution_urgency: float = Field(default=0.5, ge=0.0, le=1.0, description="Execution urgency")
    market_impact_estimate: Optional[float] = Field(None, description="Estimated market impact")
    
    @validator("consensus_signal", pre=True, always=True)
    def determine_consensus(cls, v, values):
        """Determine consensus signal from component signals"""
        if "component_signals" in values and "agent_weights" in values:
            signals = values["component_signals"]
            weights = values["agent_weights"]
            
            # Weighted voting for consensus
            vote_scores = {
                SignalType.STRONG_BUY: 2,
                SignalType.BUY: 1,
                SignalType.HOLD: 0,
                SignalType.SELL: -1,
                SignalType.STRONG_SELL: -2
            }
            
            weighted_score = 0
            total_weight = 0
            
            for signal in signals:
                agent_weight = weights.get(signal.source.value, 1.0)
                signal_weight = signal.confidence * agent_weight
                weighted_score += vote_scores[signal.signal_type] * signal_weight
                total_weight += signal_weight
            
            if total_weight > 0:
                avg_score = weighted_score / total_weight
                
                if avg_score >= 1.5:
                    return SignalType.STRONG_BUY
                elif avg_score >= 0.5:
                    return SignalType.BUY
                elif avg_score <= -1.5:
                    return SignalType.STRONG_SELL
                elif avg_score <= -0.5:
                    return SignalType.SELL
                else:
                    return SignalType.HOLD
        
        return v or SignalType.HOLD
    
    def calculate_consensus_metrics(self) -> None:
        """Calculate agreement, uncertainty, and reliability metrics"""
        if not self.component_signals:
            return
        
        # Calculate agreement score
        signal_types = [s.signal_type for s in self.component_signals]
        unique_signals = set(signal_types)
        
        if len(unique_signals) == 1:
            self.agreement_score = 1.0
        else:
            # Count votes for most common signal
            from collections import Counter
            signal_counts = Counter(signal_types)
            max_count = max(signal_counts.values())
            self.agreement_score = max_count / len(signal_types)
        
        # Calculate uncertainty (inverse of confidence variance)
        confidences = [s.confidence for s in self.component_signals]
        if len(confidences) > 1:
            import numpy as np
            confidence_var = np.var(confidences)
            self.uncertainty = min(1.0, confidence_var * 2)  # Scale variance
        else:
            self.uncertainty = 0.0
        
        # Reliability based on component signal strengths
        strengths = [s.strength for s in self.component_signals]
        strength_scores = {
            SignalStrength.VERY_STRONG: 1.0,
            SignalStrength.STRONG: 0.75,
            SignalStrength.MODERATE: 0.5,
            SignalStrength.WEAK: 0.25
        }
        avg_strength = sum(strength_scores[s] for s in strengths) / len(strengths)
        self.reliability = avg_strength
    
    def get_supporting_agents(self) -> List[str]:
        """Get list of agents supporting the consensus"""
        supporting = []
        for signal in self.component_signals:
            if signal.signal_type == self.consensus_signal:
                supporting.append(signal.source.value)
        return supporting
    
    def get_dissenting_agents(self) -> List[str]:
        """Get list of agents dissenting from consensus"""
        dissenting = []
        for signal in self.component_signals:
            if signal.signal_type != self.consensus_signal:
                dissenting.append(signal.source.value)
        return dissenting


class SignalHistory(BaseModel):
    """Historical signal tracking for performance analysis"""
    
    signal_id: str
    symbol: str
    signal_type: SignalType
    confidence: float
    timestamp: datetime
    source: SignalSource
    
    # Outcome tracking
    executed: bool = False
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    
    # Performance metrics
    actual_return: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    duration_held: Optional[int] = None  # minutes
    
    # Final outcome
    was_profitable: Optional[bool] = None
    outcome_confidence: Optional[float] = None  # How confident we are in the outcome
    outcome_timestamp: Optional[datetime] = None 