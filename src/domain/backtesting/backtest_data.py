"""
Backtesting Data Module - Handles data fetching and caching
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
import asyncpg
import redis
import json

from src.utils.timezone_utils import now_utc, make_aware

logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Single market data point"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol
        }


class BacktestDataManager:
    """Manages data fetching and caching for backtesting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour default
        self._cache: Dict[str, pd.DataFrame] = {}
        
    async def initialize(self):
        """Initialize data connections"""
        # Initialize database pool
        if self.config.get('database_url'):
            self.db_pool = await asyncpg.create_pool(
                self.config['database_url'],
                min_size=2,
                max_size=10
            )
            
        # Initialize Redis
        if self.config.get('redis_url'):
            self.redis_client = redis.from_url(
                self.config['redis_url'],
                decode_responses=True
            )
    
    async def close(self):
        """Close connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            self.redis_client.close()
    
    async def fetch_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '5m'
    ) -> Dict[str, pd.DataFrame]:
        """Fetch market data for multiple symbols"""
        market_data = {}
        
        # Ensure timezone aware dates
        start_date = make_aware(start_date)
        end_date = make_aware(end_date)
        
        # Try to fetch from cache first
        tasks = []
        for symbol in symbols:
            cache_key = self._get_cache_key(symbol, start_date, end_date, interval)
            
            # Check memory cache
            if cache_key in self._cache:
                market_data[symbol] = self._cache[cache_key]
                continue
                
            # Check Redis cache
            if self.redis_client:
                cached_data = await self._get_from_redis(cache_key)
                if cached_data:
                    market_data[symbol] = cached_data
                    self._cache[cache_key] = cached_data
                    continue
            
            # Need to fetch from database
            tasks.append(self._fetch_symbol_data(symbol, start_date, end_date, interval))
        
        # Fetch missing data in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error fetching data: {result}")
                    continue
                    
                if result and len(result) > 0:
                    symbol = result[0].symbol
                    df = self._convert_to_dataframe(result)
                    market_data[symbol] = df
                    
                    # Cache the data
                    cache_key = self._get_cache_key(
                        symbol, start_date, end_date, interval
                    )
                    self._cache[cache_key] = df
                    
                    if self.redis_client:
                        await self._save_to_redis(cache_key, df)
        
        return market_data
    
    async def _fetch_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> List[MarketDataPoint]:
        """Fetch data for a single symbol from database"""
        if not self.db_pool:
            return self._generate_mock_data(symbol, start_date, end_date, interval)
        
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM market_data
                    WHERE symbol = $1 
                    AND timestamp >= $2 
                    AND timestamp <= $3
                    AND interval = $4
                    ORDER BY timestamp
                """
                
                rows = await conn.fetch(
                    query, symbol, start_date, end_date, interval
                )
                
                return [
                    MarketDataPoint(
                        timestamp=row['timestamp'],
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume']),
                        symbol=symbol
                    )
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Database error fetching {symbol}: {e}")
            return self._generate_mock_data(symbol, start_date, end_date, interval)
    
    def _generate_mock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> List[MarketDataPoint]:
        """Generate mock data for testing"""
        # Map interval to pandas frequency
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D'
        }
        
        freq = interval_map.get(interval, '5min')
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)
        base_price = 100 + np.random.rand() * 400
        
        data_points = []
        price = base_price
        
        for ts in timestamps:
            # Random walk with mean reversion
            returns = np.random.normal(0, 0.02)
            price *= (1 + returns)
            price = price * 0.99 + base_price * 0.01  # Mean reversion
            
            # OHLC generation
            open_price = price
            close_price = price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
            volume = np.random.lognormal(15, 1)
            
            data_points.append(MarketDataPoint(
                timestamp=ts,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=int(volume),
                symbol=symbol
            ))
            
            price = close_price
        
        return data_points
    
    def _convert_to_dataframe(self, data_points: List[MarketDataPoint]) -> pd.DataFrame:
        """Convert data points to pandas DataFrame"""
        if not data_points:
            return pd.DataFrame()
        
        data = {
            'open': [p.open for p in data_points],
            'high': [p.high for p in data_points],
            'low': [p.low for p in data_points],
            'close': [p.close for p in data_points],
            'volume': [p.volume for p in data_points]
        }
        
        index = pd.DatetimeIndex([p.timestamp for p in data_points])
        return pd.DataFrame(data, index=index)
    
    def _get_cache_key(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> str:
        """Generate cache key"""
        return f"backtest:data:{symbol}:{start_date.date()}:{end_date.date()}:{interval}"
    
    async def _get_from_redis(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from Redis cache"""
        try:
            data = self.redis_client.get(key)
            if data:
                # Deserialize from JSON
                data_dict = json.loads(data)
                df = pd.DataFrame(data_dict['data'])
                df.index = pd.DatetimeIndex(data_dict['index'])
                return df
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        return None
    
    async def _save_to_redis(self, key: str, df: pd.DataFrame):
        """Save data to Redis cache"""
        try:
            # Serialize to JSON
            data_dict = {
                'data': df.to_dict(),
                'index': df.index.tolist()
            }
            self.redis_client.setex(
                key,
                self.cache_ttl,
                json.dumps(data_dict, default=str)
            )
        except Exception as e:
            logger.error(f"Redis save error: {e}")
    
    def clear_cache(self):
        """Clear memory cache"""
        self._cache.clear()
    
    async def preload_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        intervals: List[str]
    ):
        """Preload data for multiple symbols and intervals"""
        tasks = []
        
        for interval in intervals:
            task = self.fetch_market_data(symbols, start_date, end_date, interval)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Preloaded data for {len(symbols)} symbols, {len(intervals)} intervals") 