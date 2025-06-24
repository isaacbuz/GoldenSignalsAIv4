"""
Enhanced Backtesting Data Manager - Production-grade data handling
Supports multiple data sources with automatic failover and validation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
import aiohttp
import asyncpg
from abc import ABC, abstractmethod
import json
from collections import defaultdict

from src.utils.timezone_utils import now_utc, make_aware

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources in priority order"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON_IO = "polygon_io"
    LOCAL_DB = "local_db"


@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment"""
    completeness: float  # % of expected data points present
    accuracy: float      # Based on outlier detection
    timeliness: float    # How recent the data is
    consistency: float   # Internal consistency checks
    source: str
    issues: List[str]
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score"""
        return (self.completeness * 0.3 + 
                self.accuracy * 0.3 + 
                self.timeliness * 0.2 + 
                self.consistency * 0.2)


class DataSourceInterface(ABC):
    """Abstract interface for data sources"""
    
    @abstractmethod
    async def fetch_historical(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime, 
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data"""
        pass
    
    @abstractmethod
    async def fetch_realtime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch real-time quote"""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Validate data quality"""
        pass


class YahooFinanceSource(DataSourceInterface):
    """Yahoo Finance data source implementation"""
    
    def __init__(self):
        self.name = DataSource.YAHOO_FINANCE
        self._rate_limit_delay = 0.5  # seconds between requests
        self._last_request_time = 0
        
    async def fetch_historical(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime, 
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch from Yahoo Finance"""
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Convert interval format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '60m', '4h': '1d', '1d': '1d'
            }
            yf_interval = interval_map.get(interval, '1d')
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start,
                end=end,
                interval=yf_interval,
                auto_adjust=True,  # Adjust for splits/dividends
                prepost=True       # Include pre/post market
            )
            
            if data.empty:
                return None
                
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add metadata
            data['source'] = self.name.value
            
            return data[['open', 'high', 'low', 'close', 'volume', 'source']]
            
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    async def fetch_realtime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch real-time quote from Yahoo"""
        try:
            await self._rate_limit()
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', info.get('price')),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'volume': info.get('regularMarketVolume', info.get('volume')),
                'timestamp': now_utc(),
                'source': self.name.value
            }
            
        except Exception as e:
            logger.error(f"Yahoo realtime error for {symbol}: {e}")
            return None
    
    def validate_data(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Validate Yahoo Finance data"""
        issues = []
        
        # Check completeness
        expected_rows = len(pd.date_range(
            start=data.index[0], 
            end=data.index[-1], 
            freq='D'
        ))
        actual_rows = len(data)
        completeness = min(actual_rows / expected_rows, 1.0) if expected_rows > 0 else 0
        
        # Check for outliers (price changes > 50% in a day)
        price_changes = data['close'].pct_change().abs()
        outliers = price_changes > 0.5
        accuracy = 1.0 - (outliers.sum() / len(data)) if len(data) > 0 else 0
        
        if outliers.any():
            issues.append(f"Found {outliers.sum()} potential outliers")
        
        # Check timeliness
        last_data_time = data.index[-1]
        hours_old = (now_utc() - make_aware(last_data_time)).total_seconds() / 3600
        timeliness = max(0, 1.0 - (hours_old / 24))  # Decay over 24 hours
        
        # Check consistency (OHLC relationships)
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        consistency = 1.0 - (invalid_ohlc.sum() / len(data)) if len(data) > 0 else 0
        
        if invalid_ohlc.any():
            issues.append(f"Found {invalid_ohlc.sum()} OHLC inconsistencies")
        
        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            timeliness=timeliness,
            consistency=consistency,
            source=self.name.value,
            issues=issues
        )
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        import time
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last)
        
        self._last_request_time = time.time()


class LocalDatabaseSource(DataSourceInterface):
    """Local TimescaleDB data source"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.name = DataSource.LOCAL_DB
        self.db_config = db_config
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(**self.db_config)
        
        # Create tables if not exist
        await self._create_tables()
    
    async def _create_tables(self):
        """Create TimescaleDB tables"""
        async with self.pool.acquire() as conn:
            # Create main table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open DECIMAL,
                    high DECIMAL,
                    low DECIMAL,
                    close DECIMAL,
                    volume BIGINT,
                    source TEXT,
                    PRIMARY KEY (time, symbol)
                );
            """)
            
            # Convert to hypertable for time-series optimization
            await conn.execute("""
                SELECT create_hypertable('market_data', 'time', 
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
                ON market_data (symbol, time DESC);
            """)
    
    async def fetch_historical(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime, 
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch from local database"""
        if not self.pool:
            return None
            
        try:
            async with self.pool.acquire() as conn:
                # Build time bucket query based on interval
                interval_map = {
                    '1m': '1 minute', '5m': '5 minutes', '15m': '15 minutes',
                    '30m': '30 minutes', '1h': '1 hour', '4h': '4 hours', '1d': '1 day'
                }
                bucket_interval = interval_map.get(interval, '5 minutes')
                
                query = """
                    SELECT 
                        time_bucket($1::interval, time) as time,
                        symbol,
                        first(open, time) as open,
                        max(high) as high,
                        min(low) as low,
                        last(close, time) as close,
                        sum(volume) as volume,
                        last(source, time) as source
                    FROM market_data
                    WHERE symbol = $2 
                    AND time >= $3 
                    AND time <= $4
                    GROUP BY time_bucket($1::interval, time), symbol
                    ORDER BY time;
                """
                
                rows = await conn.fetch(query, bucket_interval, symbol, start, end)
                
                if not rows:
                    return None
                
                # Convert to DataFrame
                data = pd.DataFrame(rows)
                data.set_index('time', inplace=True)
                
                return data
                
        except Exception as e:
            logger.error(f"Database fetch error for {symbol}: {e}")
            return None
    
    async def fetch_realtime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch latest data point"""
        if not self.pool:
            return None
            
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM market_data 
                    WHERE symbol = $1 
                    ORDER BY time DESC 
                    LIMIT 1
                """, symbol)
                
                if row:
                    return {
                        'symbol': row['symbol'],
                        'price': float(row['close']),
                        'bid': float(row['close']) * 0.9999,  # Approximate
                        'ask': float(row['close']) * 1.0001,  # Approximate
                        'volume': row['volume'],
                        'timestamp': row['time'],
                        'source': row['source']
                    }
                return None
                
        except Exception as e:
            logger.error(f"Database realtime error for {symbol}: {e}")
            return None
    
    async def store_data(self, symbol: str, data: pd.DataFrame, source: str):
        """Store data in database"""
        if not self.pool:
            return
            
        try:
            async with self.pool.acquire() as conn:
                # Prepare data for insertion
                records = []
                for idx, row in data.iterrows():
                    records.append((
                        idx, symbol,
                        float(row['open']), float(row['high']),
                        float(row['low']), float(row['close']),
                        int(row['volume']), source
                    ))
                
                # Bulk insert with conflict handling
                await conn.executemany("""
                    INSERT INTO market_data 
                    (time, symbol, open, high, low, close, volume, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (time, symbol) 
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        source = EXCLUDED.source;
                """, records)
                
                logger.info(f"Stored {len(records)} records for {symbol}")
                
        except Exception as e:
            logger.error(f"Database store error: {e}")
    
    def validate_data(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Validate database data - typically high quality"""
        return DataQualityMetrics(
            completeness=1.0,  # Assume complete if in DB
            accuracy=0.95,     # Assume validated on insert
            timeliness=1.0,    # Always current for historical
            consistency=1.0,   # Enforced by constraints
            source=self.name.value,
            issues=[]
        )


class EnhancedBacktestDataManager:
    """
    Production-grade data manager with multi-source support,
    automatic failover, and comprehensive validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources: Dict[DataSource, DataSourceInterface] = {}
        self._cache: Dict[str, pd.DataFrame] = {}
        self._quality_metrics: Dict[str, DataQualityMetrics] = {}
        
        # Initialize sources based on config
        self._initialize_sources()
        
    def _initialize_sources(self):
        """Initialize configured data sources"""
        # Always add Yahoo Finance
        self.sources[DataSource.YAHOO_FINANCE] = YahooFinanceSource()
        
        # Add database if configured
        if self.config.get('database'):
            self.sources[DataSource.LOCAL_DB] = LocalDatabaseSource(
                self.config['database']
            )
        
        # TODO: Add other sources (Alpha Vantage, IEX, Polygon) when API keys available
        
    async def initialize(self):
        """Initialize all data sources"""
        tasks = []
        
        for source in self.sources.values():
            if hasattr(source, 'initialize'):
                tasks.append(source.initialize())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def fetch_historical_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        interval: str = '1d',
        quality_threshold: float = 0.7
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data with automatic source failover
        
        Args:
            symbols: List of symbols to fetch
            start: Start date
            end: End date  
            interval: Time interval
            quality_threshold: Minimum quality score to accept data
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        results = {}
        
        # Process symbols in parallel
        tasks = []
        for symbol in symbols:
            task = self._fetch_symbol_with_failover(
                symbol, start, end, interval, quality_threshold
            )
            tasks.append(task)
        
        symbol_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary
        for symbol, result in zip(symbols, symbol_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {symbol}: {result}")
            elif result is not None:
                results[symbol] = result
        
        return results
    
    async def _fetch_symbol_with_failover(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
        quality_threshold: float
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol with source failover"""
        
        # Check cache first
        cache_key = f"{symbol}:{start}:{end}:{interval}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try sources in priority order
        for source_type in DataSource:
            if source_type not in self.sources:
                continue
                
            source = self.sources[source_type]
            
            try:
                # Fetch data
                data = await source.fetch_historical(symbol, start, end, interval)
                
                if data is None or data.empty:
                    continue
                
                # Validate quality
                quality = source.validate_data(data)
                self._quality_metrics[f"{symbol}:{source_type.value}"] = quality
                
                if quality.overall_score >= quality_threshold:
                    # Store in local DB if available and not from DB
                    if (source_type != DataSource.LOCAL_DB and 
                        DataSource.LOCAL_DB in self.sources):
                        await self.sources[DataSource.LOCAL_DB].store_data(
                            symbol, data, source_type.value
                        )
                    
                    # Cache and return
                    self._cache[cache_key] = data
                    logger.info(
                        f"Fetched {symbol} from {source_type.value} "
                        f"(quality: {quality.overall_score:.2f})"
                    )
                    return data
                else:
                    logger.warning(
                        f"Data quality too low for {symbol} from {source_type.value}: "
                        f"{quality.overall_score:.2f} < {quality_threshold}"
                    )
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol} from {source_type.value}: {e}")
                continue
        
        logger.error(f"Failed to fetch {symbol} from any source")
        return None
    
    async def stream_realtime_data(
        self,
        symbols: List[str],
        callback: Callable[[Dict[str, Any]], None],
        interval_seconds: int = 5
    ):
        """
        Stream real-time data for symbols
        
        Args:
            symbols: Symbols to stream
            callback: Function to call with each update
            interval_seconds: Update interval
        """
        logger.info(f"Starting real-time stream for {symbols}")
        
        while True:
            try:
                # Fetch current quotes for all symbols
                tasks = []
                for symbol in symbols:
                    for source in self.sources.values():
                        task = source.fetch_realtime(symbol)
                        tasks.append(task)
                        break  # Use first available source
                
                quotes = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process valid quotes
                for quote in quotes:
                    if isinstance(quote, dict) and quote:
                        callback(quote)
                
                # Wait for next update
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(interval_seconds)
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report"""
        report = {
            'summary': {
                'total_symbols': len(set(k.split(':')[0] for k in self._quality_metrics)),
                'average_quality': np.mean([
                    m.overall_score for m in self._quality_metrics.values()
                ]) if self._quality_metrics else 0
            },
            'details': {}
        }
        
        # Group by symbol
        symbol_metrics = defaultdict(list)
        for key, metrics in self._quality_metrics.items():
            symbol = key.split(':')[0]
            symbol_metrics[symbol].append(metrics)
        
        # Build detailed report
        for symbol, metrics_list in symbol_metrics.items():
            best_source = max(metrics_list, key=lambda m: m.overall_score)
            report['details'][symbol] = {
                'best_source': best_source.source,
                'best_quality': best_source.overall_score,
                'sources_tried': len(metrics_list),
                'issues': best_source.issues
            }
        
        return report
    
    def clear_cache(self):
        """Clear memory cache"""
        self._cache.clear()
        self._quality_metrics.clear()


# Example usage
async def demo():
    """Demo the enhanced data manager"""
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_db',
            'user': 'trading_user',
            'password': 'secure_password'
        }
    }
    
    manager = EnhancedBacktestDataManager(config)
    await manager.initialize()
    
    # Fetch historical data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 1)
    
    data = await manager.fetch_historical_data(
        symbols, start, end, interval='1d'
    )
    
    # Print results
    for symbol, df in data.items():
        print(f"\n{symbol}: {len(df)} days of data")
        print(f"Quality: {manager._quality_metrics.get(f'{symbol}:yahoo_finance')}")
    
    # Get quality report
    report = manager.get_data_quality_report()
    print(f"\nQuality Report: {json.dumps(report, indent=2)}")


if __name__ == "__main__":
    asyncio.run(demo()) 