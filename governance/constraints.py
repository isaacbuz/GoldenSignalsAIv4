# governance/constraints.py
# Purpose: Enforces ethical and regulatory constraints on trading actions, such as
# position size limits, trade frequency, and restricted symbols. Ensures compliance
# for options trading by limiting exposure and preventing unauthorized trades.

import logging
from datetime import datetime
from typing import Dict

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class TradingConstraints:
    """Enforces ethical and regulatory constraints on trading actions."""

    def __init__(self):
        """Initialize with predefined rules."""
        self.max_position_size = 0.3  # Max 30% of portfolio in a single position
        self.max_daily_trades = 10  # Max trades per day
        self.restricted_symbols = ["XYZ"]  # Example restricted symbols
        self.trade_count = 0
        self.last_reset = datetime.now()
        logger.info({"message": "TradingConstraints initialized"})

    def reset_daily_trades(self):
        """Reset the daily trade counter if a new day has started."""
        logger.debug({"message": "Checking daily trade counter reset"})
        try:
            now = datetime.now()
            if now.date() > self.last_reset.date():
                self.trade_count = 0
                self.last_reset = now
                logger.info({"message": "Daily trade counter reset"})
        except Exception as e:
            logger.error({"message": f"Failed to reset daily trade counter: {str(e)}"})

    def check_position_size(self, trade: Dict, portfolio_value: float) -> bool:
        """Check if the trade adheres to position size limits.

        Args:
            trade (Dict): Trade details with 'size', 'symbol', 'price'.
            portfolio_value (float): Current portfolio value.

        Returns:
            bool: True if within limits, False otherwise.
        """
        logger.debug({"message": f"Checking position size for trade: {trade}"})
        try:
            position_value = trade["size"] * trade.get("price", 1.0)
            position_ratio = (
                position_value / portfolio_value if portfolio_value > 0 else 0
            )
            if position_ratio > self.max_position_size:
                logger.warning(
                    {
                        "message": f"Position size exceeds limit: {position_ratio:.2f} > {self.max_position_size}",
                        "symbol": trade["symbol"],
                    }
                )
                return False
            logger.debug(
                {"message": f"Position size within limit: {position_ratio:.2f}"}
            )
            return True
        except Exception as e:
            logger.error({"message": f"Failed to check position size: {str(e)}"})
            return False

    def check_trade_frequency(self) -> bool:
        """Check if the trade frequency limit has been exceeded.

        Returns:
            bool: True if within limits, False otherwise.
        """
        logger.debug(
            {"message": f"Checking trade frequency: {self.trade_count} trades"}
        )
        try:
            self.reset_daily_trades()
            if self.trade_count >= self.max_daily_trades:
                logger.warning(
                    {
                        "message": f"Daily trade limit exceeded: {self.trade_count} >= {self.max_daily_trades}"
                    }
                )
                return False
            self.trade_count += 1
            logger.debug({"message": f"Trade count incremented to {self.trade_count}"})
            return True
        except Exception as e:
            logger.error({"message": f"Failed to check trade frequency: {str(e)}"})
            return False

    def check_restrictions(self, trade: Dict) -> bool:
        """Check if the trade violates any restrictions (e.g., restricted symbols).

        Args:
            trade (Dict): Trade details with 'symbol'.

        Returns:
            bool: True if allowed, False if restricted.
        """
        logger.debug({"message": f"Checking restrictions for trade: {trade}"})
        try:
            symbol = trade["symbol"]
            if symbol in self.restricted_symbols:
                logger.warning({"message": f"Trade on restricted symbol: {symbol}"})
                return False
            logger.debug({"message": f"No restrictions for symbol: {symbol}"})
            return True
        except Exception as e:
            logger.error({"message": f"Failed to check restrictions: {str(e)}"})
            return False

    def enforce(self, trade: Dict, portfolio_value: float) -> bool:
        """Enforce all trading constraints for options trading compliance.

        Args:
            trade (Dict): Trade details with 'symbol', 'size', 'price'.
            portfolio_value (float): Current portfolio value.

        Returns:
            bool: True if trade is allowed, False otherwise.
        """
        logger.info({"message": f"Enforcing constraints for trade: {trade}"})
        try:
            checks = [
                self.check_position_size(trade, portfolio_value),
                self.check_trade_frequency(),
                self.check_restrictions(trade),
            ]
            allowed = all(checks)
            if not allowed:
                logger.warning(
                    {"message": f"Trade rejected due to constraint violation: {trade}"}
                )
            else:
                logger.info({"message": f"Trade approved: {trade}"})
            return allowed
        except Exception as e:
            logger.error({"message": f"Failed to enforce constraints: {str(e)}"})
            return False
