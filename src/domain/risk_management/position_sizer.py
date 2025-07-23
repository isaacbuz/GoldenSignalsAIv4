"""
Position Sizer: Calculates optimal position sizes based on signal confidence and risk metrics.
"""
from typing import Dict

import numpy as np


class PositionSizer:
    def __init__(self):
        self.max_position_size = 0.1  # Maximum 10% of account per trade
        self.min_position_size = 0.01  # Minimum 1% of account per trade
        self.max_risk_per_trade = 0.02  # Maximum 2% risk per trade

    def calculate_position(self, signal: Dict, account_size: float, risk_metrics: Dict) -> Dict:
        """
        Calculate optimal position size based on signal confidence and risk metrics.
        
        Args:
            signal: Dictionary containing signal information
            account_size: Total account size in base currency
            risk_metrics: Dictionary containing risk metrics
            
        Returns:
            Dictionary containing position sizing information
        """
        # Calculate base position size from signal confidence
        confidence = signal.get("confidence", 0.0)
        base_size = self._calculate_base_size(confidence)
        
        # Adjust for volatility
        volatility = risk_metrics.get("volatility", 0.0)
        vol_adjusted_size = self._adjust_for_volatility(base_size, volatility)
        
        # Adjust for correlation
        correlation = risk_metrics.get("correlation", 0.0)
        corr_adjusted_size = self._adjust_for_correlation(vol_adjusted_size, correlation)
        
        # Apply risk limits
        max_loss = risk_metrics.get("max_loss", 0.0)
        risk_adjusted_size = self._apply_risk_limits(corr_adjusted_size, max_loss, account_size)
        
        # Calculate final position size
        final_size = self._calculate_final_size(risk_adjusted_size, account_size)
        
        return {
            "position_size": final_size,
            "position_value": final_size * account_size,
            "risk_metrics": {
                "base_size": base_size,
                "volatility_adjustment": vol_adjusted_size,
                "correlation_adjustment": corr_adjusted_size,
                "risk_adjustment": risk_adjusted_size
            }
        }

    def _calculate_base_size(self, confidence: float) -> float:
        """Calculate base position size from signal confidence."""
        # Linear scaling from min to max based on confidence
        return self.min_position_size + (self.max_position_size - self.min_position_size) * confidence

    def _adjust_for_volatility(self, base_size: float, volatility: float) -> float:
        """Adjust position size based on volatility."""
        # Reduce position size as volatility increases
        vol_factor = 1.0 / (1.0 + volatility)
        return base_size * vol_factor

    def _adjust_for_correlation(self, size: float, correlation: float) -> float:
        """Adjust position size based on correlation with existing positions."""
        # Reduce position size as correlation increases
        corr_factor = 1.0 - abs(correlation)
        return size * corr_factor

    def _apply_risk_limits(self, size: float, max_loss: float, account_size: float) -> float:
        """Apply risk limits to position size."""
        # Calculate maximum position size based on risk limit
        max_risk_size = (self.max_risk_per_trade * account_size) / max_loss
        
        # Take the minimum of the calculated size and risk-limited size
        return min(size, max_risk_size)

    def _calculate_final_size(self, adjusted_size: float, account_size: float) -> float:
        """Calculate final position size with account size limits."""
        # Ensure position size is within min/max limits
        final_size = np.clip(adjusted_size, self.min_position_size, self.max_position_size)
        
        # Round to 4 decimal places
        return round(final_size, 4) 