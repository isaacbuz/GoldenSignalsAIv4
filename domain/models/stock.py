# domain/models/stock.py
# Purpose: Defines a Pydantic model for validating stock data, ensuring data integrity
# for trading signals and options analysis. Split from data_models.py for modularity.

import logging

from pydantic import BaseModel, Field

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class StockData(BaseModel):
    """Pydantic model for stock data validation."""

    timestamp: str = Field(..., alias="date", description="Timestamp of the data point")
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Trading volume")

    class Config:
        """Pydantic configuration for alias support."""

        allow_population_by_field_name = True

    def validate_data(self):
        """Validate the stock data instance.

        Returns:
            bool: True if valid, False otherwise.
        """
        logger.debug({"message": f"Validating stock data for {self.symbol}"})
        try:
            self.validate()
            logger.debug(
                {"message": f"Stock data validated successfully for {self.symbol}"}
            )
            return True
        except Exception as e:
            logger.error(
                {"message": f"Stock data validation failed for {self.symbol}: {str(e)}"}
            )
            return False
