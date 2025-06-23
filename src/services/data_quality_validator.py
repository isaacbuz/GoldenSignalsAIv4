"""
Data Quality Validator Service for GoldenSignalsAI V2
Handles data validation, fallback sources, and retry logic
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asy, timezonencio
from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf
import aiohttp
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    """Report of data quality checks"""
    symbol: str
    is_valid: bool
    completeness: float  # 0-1 score
    accuracy: float  # 0-1 score
    timeliness: float  # 0-1 score
    consistency: float  # 0-1 score
    issues: List[str]
    recommendations: List[str]
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score"""
        return np.mean([self.completeness, self.accuracy, self.timeliness, self.consistency])


class DataQualityValidator:
    """Validates market data quality and provides fallback mechanisms"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        
        # Data source priorities
        self.data_sources = [
            self._fetch_yahoo_finance,
            self._fetch_alpha_vantage,
            self._fetch_polygon,
            self._fetch_finnhub
        ]
        
    async def get_market_data_with_fallback(self, symbol: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Fetch market data with fallback to multiple sources
        Returns: (data, source_name)
        """
        for i, source_func in enumerate(self.data_sources):
            try:
                logger.info(f"Attempting to fetch {symbol} from source {i+1}/{len(self.data_sources)}")
                data = await source_func(symbol)
                if data is not None and not data.empty:
                    source_name = source_func.__name__.replace('_fetch_', '')
                    logger.info(f"✅ Successfully fetched {symbol} from {source_name}")
                    return data, source_name
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} from source {i+1}: {e}")
                continue
                
        logger.error(f"❌ Failed to fetch {symbol} from all sources")
        return None, "none"
    
    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _fetch_yahoo_finance(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance with retry logic"""
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            
            # Try to get historical data
            data = await loop.run_in_executor(
                None,
                lambda: ticker.history(period=period, interval="1d")
            )
            
            if data.empty:
                # Try with a shorter period
                data = await loop.run_in_executor(
                    None,
                    lambda: ticker.history(period="5d", interval="1d")
                )
            
            # Ensure index is timezone-aware and in UTC
            if not data.empty:
                if data.index.tz is None:
                    # If naive, localize to UTC
                    data.index = data.index.tz_localize('UTC')
                else:
                    # If already has timezone, convert to UTC
                    data.index = data.index.tz_convert('UTC')
            
            return data
            
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            raise
            
    async def _fetch_alpha_vantage(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage"""
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not configured")
            return None
            
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.alpha_vantage_key,
            "outputsize": "compact"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "Time Series (Daily)" in data:
                        # Convert to DataFrame
                        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        df.index = pd.to_datetime(df.index)
                        df = df.astype(float)
                        return df.sort_index()
                    elif "Note" in data:
                        logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                    elif "Error Message" in data:
                        logger.error(f"Alpha Vantage error: {data['Error Message']}")
                        
        return None
        
    async def _fetch_polygon(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io"""
        if not self.polygon_key:
            logger.warning("Polygon API key not configured")
            return None
            
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params={"apiKey": self.polygon_key}) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "OK" and "results" in data:
                        # Convert to DataFrame
                        df = pd.DataFrame(data["results"])
                        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        df.rename(columns={
                            'o': 'Open',
                            'h': 'High',
                            'l': 'Low',
                            'c': 'Close',
                            'v': 'Volume'
                        }, inplace=True)
                        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        
        return None
        
    async def _fetch_finnhub(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Finnhub"""
        if not self.finnhub_key:
            logger.warning("Finnhub API key not configured")
            return None
            
        # Implementation for Finnhub
        # This is a placeholder - actual implementation would depend on Finnhub API
        return None
        
    def validate_market_data(self, data: pd.DataFrame, symbol: str) -> DataQualityReport:
        """Validate the quality of market data"""
        issues = []
        recommendations = []
        
        # Completeness check
        completeness = self._check_completeness(data, issues, recommendations)
        
        # Accuracy check
        accuracy = self._check_accuracy(data, issues, recommendations)
        
        # Timeliness check
        timeliness = self._check_timeliness(data, issues, recommendations)
        
        # Consistency check
        consistency = self._check_consistency(data, issues, recommendations)
        
        # Overall validity
        is_valid = (completeness >= 0.8 and accuracy >= 0.8 and 
                   timeliness >= 0.8 and consistency >= 0.8)
        
        return DataQualityReport(
            symbol=symbol,
            is_valid=is_valid,
            completeness=completeness,
            accuracy=accuracy,
            timeliness=timeliness,
            consistency=consistency,
            issues=issues,
            recommendations=recommendations
        )
        
    def _check_completeness(self, data: pd.DataFrame, issues: List[str], 
                          recommendations: List[str]) -> float:
        """Check data completeness"""
        total_expected = len(data)
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            issues.append(f"Missing values detected: {dict(missing_counts[missing_counts > 0])}")
            recommendations.append("Consider interpolation or forward-fill for missing values")
            
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            recommendations.append("Use a different data source that provides all required fields")
            
        completeness_score = 1.0 - (total_missing / (total_expected * len(data.columns)))
        return max(0, completeness_score)
        
    def _check_accuracy(self, data: pd.DataFrame, issues: List[str], 
                       recommendations: List[str]) -> float:
        """Check data accuracy"""
        accuracy_score = 1.0
        
        # Check for outliers using IQR method
        for column in ['Open', 'High', 'Low', 'Close']:
            if column in data.columns:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[column] < (Q1 - 3 * IQR)) | 
                           (data[column] > (Q3 + 3 * IQR))).sum()
                
                if outliers > 0:
                    issues.append(f"{outliers} potential outliers in {column}")
                    accuracy_score *= 0.9
                    
        # Check for logical consistency
        invalid_hl = (data['High'] < data['Low']).sum() if 'High' in data.columns and 'Low' in data.columns else 0
        if invalid_hl > 0:
            issues.append(f"{invalid_hl} rows where High < Low")
            recommendations.append("Remove or correct invalid price relationships")
            accuracy_score *= 0.8
            
        return accuracy_score
        
    def _check_timeliness(self, data: pd.DataFrame, issues: List[str], 
                         recommendations: List[str]) -> float:
        """Check data timeliness"""
        if data.empty:
            return 0.0
            
        latest_date = data.index.max()
        current_date = pd.Timestamp.now(tz='UTC')
        
        # Check if data is recent (within last trading day)
        days_old = (current_date - latest_date).days
        
        if days_old > 1:
            issues.append(f"Data is {days_old} days old")
            recommendations.append("Fetch more recent data or check if market is closed")
            
        # Score based on age
        if days_old == 0:
            return 1.0
        elif days_old == 1:
            return 0.9
        elif days_old <= 3:
            return 0.7
        else:
            return max(0, 1.0 - (days_old / 10))
            
    def _check_consistency(self, data: pd.DataFrame, issues: List[str], 
                          recommendations: List[str]) -> float:
        """Check data consistency"""
        consistency_score = 1.0
        
        # Check for duplicate timestamps
        duplicates = data.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate timestamps found")
            recommendations.append("Remove duplicate entries")
            consistency_score *= 0.9
            
        # Check for gaps in time series
        if len(data) > 1:
            expected_days = pd.bdate_range(start=data.index.min(), end=data.index.max())
            missing_days = len(expected_days) - len(data)
            
            if missing_days > 0:
                gap_ratio = missing_days / len(expected_days)
                if gap_ratio > 0.1:  # More than 10% missing
                    issues.append(f"{missing_days} missing trading days ({gap_ratio:.1%})")
                    recommendations.append("Consider using a different data source with complete history")
                consistency_score *= (1 - gap_ratio)
                
        return consistency_score
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess market data"""
        cleaned = data.copy()
        
        # Remove duplicates
        cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
        
        # Sort by index
        cleaned = cleaned.sort_index()
        
        # Forward fill missing values (max 2 days)
        cleaned = cleaned.fillna(method='ffill', limit=2)
        
        # Remove remaining NaN rows
        cleaned = cleaned.dropna()
        
        # Ensure positive values
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].abs()
                
        # Fix High/Low relationships
        if 'High' in cleaned.columns and 'Low' in cleaned.columns:
            cleaned['High'] = cleaned[['High', 'Low', 'Open', 'Close']].max(axis=1)
            cleaned['Low'] = cleaned[['High', 'Low', 'Open', 'Close']].min(axis=1)
            
        return cleaned
        
    def detect_outliers(self, data: pd.DataFrame, column: str = 'Close', 
                       method: str = 'zscore', threshold: float = 3.0) -> List[int]:
        """Detect outliers in the data"""
        if column not in data.columns:
            return []
            
        if method == 'zscore':
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            return data.index[z_scores > threshold].tolist()
            
        elif method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (data[column] < (Q1 - threshold * IQR)) | (data[column] > (Q3 + threshold * IQR))
            return data.index[outliers].tolist()
            
        return []
        
    def normalize_features(self, data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """Normalize features for model input"""
        normalized = data.copy()
        
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        columns_to_normalize = [col for col in numeric_columns if col in data.columns]
        
        if method == 'minmax':
            for col in columns_to_normalize:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    normalized[col] = (data[col] - min_val) / (max_val - min_val)
                    
        elif method == 'zscore':
            for col in columns_to_normalize:
                mean = data[col].mean()
                std = data[col].std()
                if std > 0:
                    normalized[col] = (data[col] - mean) / std
                    
        return normalized 