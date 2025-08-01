"""
Validation utilities for market data and time series.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def validate_market_data(data: Dict[str, Any], required_fields: Optional[List[str]] = None) -> bool:
    """Validate market data dictionary has required fields and proper types.

    Args:
        data (Dict[str, Any]): Market data dictionary to validate.
        required_fields (List[str], optional): List of required field names.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(data, dict):
        logger.error("Market data must be a dictionary")
        return False

    # Default required fields if none specified
    required_fields = required_fields or [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    # Check required fields exist
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
        return False

    # Validate numeric fields
    numeric_fields = ["open", "high", "low", "close", "volume"]
    for field in numeric_fields:
        if field in data and not isinstance(data[field], (int, float)):
            logger.error(f"Field {field} must be numeric")
            return False

    # Validate timestamp
    if "timestamp" in data and not isinstance(data["timestamp"], (int, float, str)):
        logger.error("Timestamp must be numeric or string")
        return False

    return True

def validate_time_series(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_periods: int = 2
) -> bool:
    """Validate time series data meets requirements.

    Args:
        data (pd.DataFrame): Time series data to validate.
        required_columns (List[str], optional): Required column names.
        min_periods (int): Minimum number of periods required.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(data, pd.DataFrame):
        logger.error("Time series data must be a pandas DataFrame")
        return False

    # Check minimum periods
    if len(data) < min_periods:
        logger.error(f"Time series must have at least {min_periods} periods")
        return False

    # Default required columns if none specified
    required_columns = required_columns or [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    # Check required columns exist
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False

    # Check for missing values
    na_columns = data[required_columns].columns[data[required_columns].isna().any()].tolist()
    if na_columns:
        logger.error(f"Missing values in columns: {na_columns}")
        return False

    # Check numeric columns
    numeric_columns = ["open", "high", "low", "close", "volume"]
    non_numeric = [
        col for col in numeric_columns
        if col in data.columns and not np.issubdtype(data[col].dtype, np.number)
    ]
    if non_numeric:
        logger.error(f"Non-numeric data in columns: {non_numeric}")
        return False

    # Check for duplicate indices
    if data.index.duplicated().any():
        logger.error("Time series contains duplicate indices")
        return False

    return True

def validate_ohlcv(data: pd.DataFrame) -> bool:
    """Validate OHLCV data meets trading requirements.

    Args:
        data (pd.DataFrame): OHLCV data to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not validate_time_series(data):
        return False

    # Validate price relationships
    invalid_rows = (
        (data["high"] < data["low"]) |
        (data["open"] < data["low"]) |
        (data["open"] > data["high"]) |
        (data["close"] < data["low"]) |
        (data["close"] > data["high"])
    )

    if invalid_rows.any():
        logger.error(f"Invalid OHLC relationships at indices: {data.index[invalid_rows].tolist()}")
        return False

    # Validate volume
    if (data["volume"] < 0).any():
        logger.error("Negative volume values found")
        return False

    return True

def validate_features(features: pd.DataFrame, required_features: List[str]) -> bool:
    """
    Validate feature DataFrame has required columns and no missing values.

    Args:
        features: Feature DataFrame
        required_features: List of required feature names

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required features exist
        if not all(f in features.columns for f in required_features):
            return False

        # Check for missing values
        if features[required_features].isnull().any().any():
            return False

        return True

    except Exception:
        return False
