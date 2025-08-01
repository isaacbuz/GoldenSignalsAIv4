"""
Position Sizing Service with Kelly Criterion
Optimal bet sizing for trading signals based on win probability and payoff ratio
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""

    recommended_size: float  # As percentage of capital
    kelly_percentage: float  # Raw Kelly percentage
    adjusted_size: float  # After applying safety factors
    risk_amount: float  # Dollar risk for the position
    shares: int  # Number of shares to trade
    reasoning: str  # Explanation of calculation


class PositionSizingService:
    """
    Position sizing using Kelly Criterion and risk management rules

    Kelly Formula: f = (p * b - q) / b
    where:
    - f = fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = payoff ratio (win amount / loss amount)
    """

    def __init__(
        self,
        max_position_pct: float = 0.25,  # Max 25% of capital per position
        kelly_fraction: float = 0.25,  # Use 25% of Kelly (conservative)
        min_position_pct: float = 0.01,  # Min 1% position
        max_risk_per_trade: float = 0.02,  # Max 2% risk per trade
    ):
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.min_position_pct = min_position_pct
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_kelly_percentage(
        self, win_probability: float, win_amount: float, loss_amount: float
    ) -> float:
        """
        Calculate raw Kelly percentage

        Args:
            win_probability: Probability of winning (0-1)
            win_amount: Expected profit if win
            loss_amount: Expected loss if lose

        Returns:
            Kelly percentage (can be negative)
        """
        if loss_amount <= 0:
            logger.warning("Loss amount must be positive")
            return 0.0

        # Payoff ratio
        b = win_amount / loss_amount

        # Probability of losing
        q = 1 - win_probability

        # Kelly formula
        kelly = (win_probability * b - q) / b

        return kelly

    def calculate_position_size(
        self,
        capital: float,
        signal_confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        historical_win_rate: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size for a trade

        Args:
            capital: Total available capital
            signal_confidence: AI signal confidence (0-1)
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            take_profit: Take profit target
            historical_win_rate: Historical win rate for similar signals
            volatility: Current market volatility

        Returns:
            PositionSizeResult with sizing details
        """
        # Calculate risk/reward
        risk_per_share = abs(entry_price - stop_loss)
        reward_per_share = abs(take_profit - entry_price)

        if risk_per_share <= 0:
            return PositionSizeResult(
                recommended_size=self.min_position_pct,
                kelly_percentage=0,
                adjusted_size=self.min_position_pct,
                risk_amount=capital * self.min_position_pct,
                shares=int((capital * self.min_position_pct) / entry_price),
                reasoning="Invalid stop loss - using minimum position size",
            )

        # Payoff ratio
        payoff_ratio = reward_per_share / risk_per_share

        # Estimate win probability
        win_probability = self._estimate_win_probability(
            signal_confidence, historical_win_rate, payoff_ratio, volatility
        )

        # Calculate raw Kelly percentage
        kelly_pct = self.calculate_kelly_percentage(
            win_probability, reward_per_share, risk_per_share
        )

        # Apply Kelly fraction (conservative approach)
        adjusted_kelly = kelly_pct * self.kelly_fraction

        # Apply position limits
        position_pct = max(self.min_position_pct, min(adjusted_kelly, self.max_position_pct))

        # Apply risk-based position sizing
        risk_based_size = self._calculate_risk_based_size(capital, entry_price, stop_loss)

        # Use the more conservative of Kelly and risk-based
        final_position_pct = min(position_pct, risk_based_size)

        # Calculate share count
        position_value = capital * final_position_pct
        shares = int(position_value / entry_price)

        # Actual risk amount
        risk_amount = shares * risk_per_share

        # Generate reasoning
        reasoning = self._generate_reasoning(
            kelly_pct, win_probability, payoff_ratio, final_position_pct
        )

        return PositionSizeResult(
            recommended_size=final_position_pct,
            kelly_percentage=kelly_pct,
            adjusted_size=adjusted_kelly,
            risk_amount=risk_amount,
            shares=shares,
            reasoning=reasoning,
        )

    def _estimate_win_probability(
        self,
        signal_confidence: float,
        historical_win_rate: Optional[float],
        payoff_ratio: float,
        volatility: Optional[float],
    ) -> float:
        """
        Estimate win probability based on multiple factors

        Args:
            signal_confidence: AI confidence (0-1)
            historical_win_rate: Historical performance
            payoff_ratio: Risk/reward ratio
            volatility: Market volatility

        Returns:
            Estimated win probability (0-1)
        """
        # Base probability from signal confidence
        base_prob = signal_confidence * 0.6  # Conservative adjustment

        # Adjust for historical performance
        if historical_win_rate is not None:
            base_prob = (base_prob + historical_win_rate) / 2

        # Adjust for payoff ratio
        # Higher payoff ratios typically have lower win rates
        if payoff_ratio > 3:
            base_prob *= 0.8
        elif payoff_ratio > 2:
            base_prob *= 0.9

        # Adjust for volatility
        if volatility is not None:
            if volatility > 0.03:  # High volatility (>3%)
                base_prob *= 0.9
            elif volatility < 0.01:  # Low volatility (<1%)
                base_prob *= 1.1

        # Ensure probability is in valid range
        return max(0.1, min(0.9, base_prob))

    def _calculate_risk_based_size(
        self, capital: float, entry_price: float, stop_loss: float
    ) -> float:
        """
        Calculate position size based on risk management rules

        Args:
            capital: Total capital
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Position size as percentage of capital
        """
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)

        # Maximum risk amount
        max_risk_amount = capital * self.max_risk_per_trade

        # Shares based on risk
        shares_by_risk = max_risk_amount / risk_per_share

        # Position value
        position_value = shares_by_risk * entry_price

        # As percentage of capital
        return position_value / capital

    def _generate_reasoning(
        self, kelly_pct: float, win_probability: float, payoff_ratio: float, final_size: float
    ) -> str:
        """Generate explanation for position sizing decision"""
        reasoning_parts = []

        # Kelly calculation
        if kelly_pct > 0:
            reasoning_parts.append(
                f"Kelly suggests {kelly_pct*100:.1f}% position "
                f"(win prob: {win_probability*100:.0f}%, "
                f"payoff: {payoff_ratio:.1f}:1)"
            )
        else:
            reasoning_parts.append("Kelly suggests no position (negative expectancy)")

        # Conservative adjustment
        reasoning_parts.append(f"Applied {self.kelly_fraction*100:.0f}% Kelly fraction for safety")

        # Final size
        reasoning_parts.append(f"Final position size: {final_size*100:.1f}% of capital")

        return ". ".join(reasoning_parts)

    def calculate_portfolio_heat(
        self, open_positions: List[Dict[str, float]], capital: float
    ) -> Dict[str, float]:
        """
        Calculate total portfolio heat (risk exposure)

        Args:
            open_positions: List of open positions with risk amounts
            capital: Total capital

        Returns:
            Portfolio heat metrics
        """
        total_risk = sum(pos.get("risk_amount", 0) for pos in open_positions)
        position_count = len(open_positions)

        return {
            "total_risk_amount": total_risk,
            "total_risk_percentage": (total_risk / capital) * 100,
            "open_positions": position_count,
            "average_risk_per_position": total_risk / position_count if position_count > 0 else 0,
            "remaining_risk_capacity": max(
                0, (capital * 0.06) - total_risk
            ),  # 6% max portfolio heat
        }

    def adjust_for_correlation(
        self,
        base_size: float,
        symbol: str,
        existing_positions: List[Dict[str, any]],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> float:
        """
        Adjust position size based on correlation with existing positions

        Args:
            base_size: Base position size
            symbol: Symbol to trade
            existing_positions: Current positions
            correlation_matrix: Correlation data

        Returns:
            Adjusted position size
        """
        if not existing_positions or not correlation_matrix:
            return base_size

        # Calculate average correlation
        correlations = []
        for pos in existing_positions:
            other_symbol = pos.get("symbol")
            if other_symbol and other_symbol in correlation_matrix.get(symbol, {}):
                correlations.append(abs(correlation_matrix[symbol][other_symbol]))

        if not correlations:
            return base_size

        avg_correlation = np.mean(correlations)

        # Reduce size for highly correlated positions
        if avg_correlation > 0.7:
            return base_size * 0.5
        elif avg_correlation > 0.5:
            return base_size * 0.75

        return base_size


# Singleton instance
_position_sizer: Optional[PositionSizingService] = None


def get_position_sizer() -> PositionSizingService:
    """Get or create position sizing service"""
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = PositionSizingService()
    return _position_sizer
