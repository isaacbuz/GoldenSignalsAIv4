# domain/models/options.py
# Purpose: Defines a Pydantic model for validating options chain data, supporting
# options trading analysis with fields for volume, open interest, and Greeks.
# Split from data_models.py for modularity and specificity.

import logging

from pydantic import BaseModel, Field

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class OptionsData(BaseModel):
    """Pydantic model for options chain data validation."""

    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    call_volume: float = Field(..., ge=0, description="Call option trading volume")
    put_volume: float = Field(..., ge=0, description="Put option trading volume")
    call_oi: float = Field(..., ge=0, description="Call option open interest")
    put_oi: float = Field(..., ge=0, description="Put option open interest")
    strike: float = Field(..., gt=0, description="Strike price")
    call_put: str = Field(..., description="Option type ('call' or 'put')")
    iv: float = Field(..., ge=0, description="Implied volatility")
    quantity: float = Field(..., ge=0, description="Number of contracts")
    delta: float = Field(None, description="Option delta (optional)")
    gamma: float = Field(None, description="Option gamma (optional)")
    theta: float = Field(None, description="Option theta (optional)")

    class Config:
        """Pydantic configuration for validation."""

        allow_population_by_field_name = True

    def validate_data(self):
        """Validate the options data instance.

        Returns:
            bool: True if valid, False otherwise.
        """
        logger.debug({"message": f"Validating options data for {self.symbol}"})
        try:
            if self.call_put not in ["call", "put"]:
                raise ValueError(f"Invalid option type: {self.call_put}")
            self.validate()
            logger.debug(
                {"message": f"Options data validated successfully for {self.symbol}"}
            )
            return True
        except Exception as e:
            logger.error(
                {
                    "message": f"Options data validation failed for {self.symbol}: {str(e)}"
                }
            )
            return False
