from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def handle_errors(func):
    """Decorator for error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

class DataManager:
    """Data management functionality integrated from AlphaPy"""
    
    def __init__(self, cache_size: int = 100):
        self.data_cache = {}
        self.cache_size = cache_size
        logger.info("Initialized DataManager with cache size %d", cache_size)
        
    def _clean_cache(self) -> None:
        """Remove oldest items if cache exceeds size limit"""
        if len(self.data_cache) > self.cache_size:
            # Remove oldest 10% of cache entries
            num_to_remove = int(self.cache_size * 0.1)
            oldest_keys = sorted(self.data_cache.keys())[:num_to_remove]
            for key in oldest_keys:
                del self.data_cache[key]
            logger.debug("Cleaned %d items from cache", num_to_remove)
    
    @handle_errors
    def fetch_market_data(
        self, 
        symbols: Union[str, List[str]], 
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Fetch market data for given symbols"""
        if isinstance(symbols, str):
            symbols = [symbols]
            
        result = {}
        for symbol in symbols:
            cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
            if cache_key not in self.data_cache:
                logger.info("Fetching data for %s from %s to %s", 
                          symbol, start_date, end_date)
                data = yf.download(symbol, start=start_date, end=end_date, 
                                 interval=interval)
                if data.empty:
                    logger.warning("No data found for %s", symbol)
                    continue
                self.data_cache[cache_key] = data
                self._clean_cache()
            result[symbol] = self.data_cache[cache_key]
            
        if not result:
            raise ValueError("No data could be fetched for any symbol")
            
        return result
    
    @handle_errors
    def prepare_features(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        lookback_periods: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling"""
        if not all(col in data.columns for col in feature_columns + [target_column]):
            missing = [col for col in feature_columns + [target_column] 
                      if col not in data.columns]
            raise ValueError(f"Missing columns in data: {missing}")
            
        features = data[feature_columns].copy()
        target = data[target_column].copy()
        
        # Create lagged features
        for col in feature_columns:
            for i in range(1, lookback_periods + 1):
                features[f"{col}_lag_{i}"] = features[col].shift(i)
                
        # Remove rows with NaN values
        features = features.dropna()
        target = target.loc[features.index]
        
        logger.info("Prepared features with shape %s", features.shape)
        return features, target
    
    @handle_errors
    def split_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets"""
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
            
        if len(features) != len(target):
            raise ValueError("Features and target must have the same length")
            
        split_idx = int(len(features) * train_ratio)
        
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]
        
        logger.info("Split data with train size %d and test size %d",
                   len(X_train), len(X_test))
        
        return X_train, X_test, y_train, y_test 