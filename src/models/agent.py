"""
Agent model for tracking AI agent performance
"""

from sqlalchemy import Column, String, Float, JSON, Integer, Boolean, DateTime
from sqlalchemy.sql import func
from .base import BaseModel


class Agent(BaseModel):
    """AI Agent model for tracking performance"""
    
    __tablename__ = "agents"
    
    # Agent identification
    name = Column(String(100), unique=True, nullable=False, index=True)
    agent_type = Column(String(50), nullable=False, index=True)  # rsi, macd, sentiment, etc.
    version = Column(String(20), default="1.0.0", nullable=False)
    description = Column(String(500), nullable=True)
    
    # Agent configuration
    config = Column(JSON, nullable=True)  # Agent-specific configuration
    parameters = Column(JSON, nullable=True)  # Model parameters
    
    # Performance metrics
    total_signals = Column(Integer, default=0, nullable=False)
    correct_signals = Column(Integer, default=0, nullable=False)
    accuracy = Column(Float, default=0.0, nullable=False)  # Percentage accuracy
    
    # Financial performance
    total_pnl = Column(Float, default=0.0, nullable=False)
    avg_pnl_per_signal = Column(Float, default=0.0, nullable=False)
    win_rate = Column(Float, default=0.0, nullable=False)  # Percentage of profitable signals
    
    # Risk metrics
    max_drawdown = Column(Float, default=0.0, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    
    # Activity tracking
    is_active = Column(Boolean, default=True, nullable=False)
    last_signal_at = Column(DateTime(timezone=True), nullable=True)
    signals_today = Column(Integer, default=0, nullable=False)
    
    # Consensus weights (updated based on performance)
    consensus_weight = Column(Float, default=1.0, nullable=False)  # Weight in consensus
    confidence_score = Column(Float, default=0.5, nullable=False)  # Agent confidence
    
    # Performance history
    performance_history = Column(JSON, nullable=True)  # Historical performance data
    recent_accuracy = Column(Float, default=0.0, nullable=False)  # Last 30 days accuracy
    
    def __repr__(self):
        return f"<Agent(name={self.name}, type={self.agent_type}, accuracy={self.accuracy:.2f}%)>"
    
    @property
    def profit_factor(self):
        """Calculate profit factor (gross profit / gross loss)"""
        if hasattr(self, '_gross_loss') and self._gross_loss != 0:
            return getattr(self, '_gross_profit', 0) / abs(self._gross_loss)
        return None
    
    @property
    def avg_win(self):
        """Calculate average winning trade"""
        winning_signals = self.correct_signals
        if winning_signals > 0 and self.total_pnl > 0:
            return self.total_pnl / winning_signals
        return 0.0
    
    @property
    def avg_loss(self):
        """Calculate average losing trade"""
        losing_signals = self.total_signals - self.correct_signals
        if losing_signals > 0 and self.total_pnl < 0:
            return abs(self.total_pnl) / losing_signals
        return 0.0
    
    def update_performance(self, signal_correct: bool, pnl: float):
        """Update agent performance metrics"""
        self.total_signals += 1
        
        if signal_correct:
            self.correct_signals += 1
        
        # Update accuracy
        self.accuracy = (self.correct_signals / self.total_signals) * 100
        
        # Update P&L
        self.total_pnl += pnl
        self.avg_pnl_per_signal = self.total_pnl / self.total_signals
        
        # Update win rate
        profitable_signals = self.correct_signals if pnl > 0 else self.correct_signals
        self.win_rate = (profitable_signals / self.total_signals) * 100
        
        # Update consensus weight based on recent performance
        self.update_consensus_weight()
        
        # Update last signal timestamp
        self.last_signal_at = func.now()
    
    def update_consensus_weight(self):
        """Update consensus weight based on performance"""
        # Base weight on accuracy and recent performance
        base_weight = self.accuracy / 100.0
        recent_weight = self.recent_accuracy / 100.0
        
        # Combine with slight bias toward recent performance
        self.consensus_weight = (base_weight * 0.7) + (recent_weight * 0.3)
        
        # Ensure weight is between 0.1 and 2.0
        self.consensus_weight = max(0.1, min(2.0, self.consensus_weight))
    
    def get_signal_confidence(self, base_confidence: float):
        """Calculate adjusted confidence based on agent performance"""
        # Adjust base confidence by agent's historical performance
        performance_multiplier = self.accuracy / 100.0
        adjusted_confidence = base_confidence * performance_multiplier
        
        # Apply consensus weight
        final_confidence = adjusted_confidence * self.consensus_weight
        
        # Ensure confidence is between 0.0 and 1.0
        return max(0.0, min(1.0, final_confidence))
    
    def to_dict(self):
        """Convert to dictionary with computed fields"""
        data = super().to_dict()
        
        # Add computed properties
        data['profit_factor'] = self.profit_factor
        data['avg_win'] = self.avg_win
        data['avg_loss'] = self.avg_loss
        
        return data 