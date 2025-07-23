"""
Risk Models - GoldenSignalsAI V3

SQLAlchemy models for risk management and tracking.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import relationship

from .base import BaseModel


class RiskMetrics(BaseModel):
    """
    Risk metrics model for tracking portfolio and position risk
    """
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String(20), nullable=True, index=True)  # Null for portfolio-level metrics
    
    # Value at Risk metrics
    var_1d = Column(Numeric(15, 2), nullable=True)      # 1-day VaR (95% confidence)
    var_5d = Column(Numeric(15, 2), nullable=True)      # 5-day VaR (95% confidence)
    var_30d = Column(Numeric(15, 2), nullable=True)     # 30-day VaR (95% confidence)
    cvar_1d = Column(Numeric(15, 2), nullable=True)     # 1-day Conditional VaR
    cvar_5d = Column(Numeric(15, 2), nullable=True)     # 5-day Conditional VaR
    cvar_30d = Column(Numeric(15, 2), nullable=True)    # 30-day Conditional VaR
    
    # Volatility metrics
    volatility_1d = Column(Numeric(8, 4), nullable=True)    # 1-day volatility
    volatility_30d = Column(Numeric(8, 4), nullable=True)   # 30-day volatility
    volatility_90d = Column(Numeric(8, 4), nullable=True)   # 90-day volatility
    volatility_1y = Column(Numeric(8, 4), nullable=True)    # 1-year volatility
    
    # Performance metrics
    sharpe_ratio = Column(Numeric(8, 4), nullable=True)
    sortino_ratio = Column(Numeric(8, 4), nullable=True)
    calmar_ratio = Column(Numeric(8, 4), nullable=True)
    max_drawdown = Column(Numeric(8, 4), nullable=True)
    max_drawdown_duration = Column(Integer, nullable=True)  # Days
    
    # Beta and correlation
    beta = Column(Numeric(8, 4), nullable=True)
    alpha = Column(Numeric(8, 4), nullable=True)
    correlation_spy = Column(Numeric(8, 4), nullable=True)  # Correlation with SPY
    correlation_qqq = Column(Numeric(8, 4), nullable=True)  # Correlation with QQQ
    
    # Concentration risk
    concentration_risk = Column(Numeric(8, 4), nullable=True)  # Herfindahl index
    sector_concentration = Column(JSON, nullable=True)         # Sector breakdown
    
    # Timestamps
    calculated_at = Column(DateTime, default=datetime.utcnow)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    def __repr__(self):
        return f"<RiskMetrics(id={self.id}, portfolio_id={self.portfolio_id}, symbol='{self.symbol}', var_1d={self.var_1d})>"


class RiskParameters(BaseModel):
    """
    Risk parameters model for configuring risk limits and thresholds
    """
    __tablename__ = "risk_parameters"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Position limits
    max_position_size = Column(Numeric(8, 4), nullable=True)      # % of portfolio
    max_sector_exposure = Column(Numeric(8, 4), nullable=True)    # % of portfolio
    max_single_stock = Column(Numeric(8, 4), nullable=True)       # % of portfolio
    
    # Risk limits
    max_portfolio_var = Column(Numeric(15, 2), nullable=True)     # Maximum VaR
    max_drawdown_limit = Column(Numeric(8, 4), nullable=True)     # Maximum drawdown %
    min_sharpe_ratio = Column(Numeric(8, 4), nullable=True)       # Minimum Sharpe ratio
    
    # Volatility limits
    max_volatility = Column(Numeric(8, 4), nullable=True)         # Maximum volatility
    max_beta = Column(Numeric(8, 4), nullable=True)               # Maximum beta
    
    # Stop loss parameters
    stop_loss_percent = Column(Numeric(8, 4), nullable=True)      # Stop loss %
    trailing_stop_percent = Column(Numeric(8, 4), nullable=True)  # Trailing stop %
    
    # Leverage limits
    max_leverage = Column(Numeric(8, 4), default=1.0)             # Maximum leverage ratio
    margin_requirement = Column(Numeric(8, 4), default=0.5)       # Margin requirement
    
    # Rebalancing parameters
    rebalance_threshold = Column(Numeric(8, 4), nullable=True)    # Rebalance trigger %
    rebalance_frequency = Column(String(20), nullable=True)       # daily, weekly, monthly
    
    # Status and timestamps
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<RiskParameters(id={self.id}, name='{self.name}', portfolio_id={self.portfolio_id})>"


class RiskAlert(BaseModel):
    """
    Risk alert model for tracking risk violations and notifications
    """
    __tablename__ = "risk_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    risk_parameter_id = Column(Integer, ForeignKey("risk_parameters.id"), nullable=True)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # var_breach, drawdown_limit, etc.
    severity = Column(String(20), nullable=False)    # low, medium, high, critical
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Risk values
    current_value = Column(Numeric(15, 4), nullable=True)
    threshold_value = Column(Numeric(15, 4), nullable=True)
    breach_percentage = Column(Numeric(8, 4), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    
    # Timestamps
    triggered_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    
    # Additional context
    symbol = Column(String(20), nullable=True)
    context_data = Column(JSON, nullable=True)  # Additional context data
    
    def __repr__(self):
        return f"<RiskAlert(id={self.id}, type='{self.alert_type}', severity='{self.severity}', active={self.is_active})>"
    
    def acknowledge(self, user: str):
        """Acknowledge the risk alert"""
        self.is_acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()
    
    def resolve(self):
        """Mark the risk alert as resolved"""
        self.is_active = False
        self.resolved_at = datetime.utcnow()


class StressTest(BaseModel):
    """
    Stress test model for scenario analysis and stress testing
    """
    __tablename__ = "stress_tests"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    
    # Test details
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    scenario_type = Column(String(50), nullable=False)  # market_crash, sector_rotation, etc.
    
    # Scenario parameters
    market_shock = Column(Numeric(8, 4), nullable=True)      # Market decline %
    volatility_shock = Column(Numeric(8, 4), nullable=True)  # Volatility increase
    correlation_shock = Column(Numeric(8, 4), nullable=True) # Correlation increase
    
    # Results
    portfolio_impact = Column(Numeric(15, 2), nullable=True)     # Portfolio value impact
    portfolio_impact_percent = Column(Numeric(8, 4), nullable=True)
    worst_position = Column(String(20), nullable=True)          # Worst performing position
    worst_position_impact = Column(Numeric(15, 2), nullable=True)
    
    # Risk metrics under stress
    stressed_var = Column(Numeric(15, 2), nullable=True)
    stressed_volatility = Column(Numeric(8, 4), nullable=True)
    stressed_sharpe = Column(Numeric(8, 4), nullable=True)
    
    # Test metadata
    positions_tested = Column(Integer, nullable=True)
    test_duration = Column(Numeric(8, 2), nullable=True)  # Test duration in seconds
    
    # Timestamps
    executed_at = Column(DateTime, default=datetime.utcnow)
    
    # Detailed results
    detailed_results = Column(JSON, nullable=True)  # Position-by-position results
    
    def __repr__(self):
        return f"<StressTest(id={self.id}, name='{self.name}', impact={self.portfolio_impact_percent}%)>" 