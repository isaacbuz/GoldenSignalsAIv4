# domain/models/signal.py
# Purpose: Defines a Pydantic model for validating trading signals, ensuring consistency
# in signal generation for options trading. Split from data_models.py for modularity.

import logging
from typing import Dict, List

from pydantic import BaseModel, Field

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class TradingSignal(BaseModel):
    """Pydantic model for trading signal validation."""

    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    action: str = Field(..., description="Trading action ('buy', 'sell', 'hold')")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0 to 1)"
    )
    ai_score: float = Field(..., description="AI prediction score")
    indicator_score: float = Field(..., description="Technical indicator score")
    final_score: float = Field(..., description="Final combined score")
    timestamp: str = Field(..., description="Timestamp of the signal")
    risk_profile: str = Field(
        ..., description="User risk profile ('conservative', 'balanced', 'aggressive')"
    )
    indicators: List[str] = Field(..., description="List of indicators used")
    metadata: Dict = Field(
        default_factory=dict, description="Additional metadata (e.g., regime)"
    )

    class Config:
        """Pydantic configuration for validation."""

        allow_population_by_field_name = True

    def validate_data(self):
        """Validate the trading signal instance.

        Returns:
            bool: True if valid, False otherwise.
        """
        logger.debug({"message": f"Validating trading signal for {self.symbol}"})
        try:
            if self.action not in ["buy", "sell", "hold"]:
                raise ValueError(f"Invalid action: {self.action}")
            if self.risk_profile not in ["conservative", "balanced", "aggressive"]:
                raise ValueError(f"Invalid risk profile: {self.risk_profile}")
            self.validate()
            logger.debug(
                {"message": f"Trading signal validated successfully for {self.symbol}"}
            )
            return True
        except Exception as e:
            logger.error(
                {
                    "message": f"Trading signal validation failed for {self.symbol}: {str(e)}"
                }
            )
            return False
