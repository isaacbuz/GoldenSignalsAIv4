"""
Live Data Connector for Model Training
Connects to PostgreSQL, Redis, and live market data sources
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database imports
import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Market data imports
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import requests

# Configuration
from config.settings import settings

# Import timezone utilities
from src.utils.timezone_utils import now_utc, make_aware, to_utc, ensure_timezone_aware

logger = logging.getLogger(__name__)


class LiveDataConnector:
    """Connects to live data sources for model training"""
    
    def __init__(self):
        self.pg_pool = None
        self.redis_client = None
        self.engine = None
        self.async_session = None
        self.alpha_vantage_ts = None
        self.alpha_vantage_ti = None
        
        # Database URLs from environment
        self.database_url = os.getenv(
            'DATABASE_URL', 
            'postgresql+asyncpg://goldensignals:password@localhost:5432/goldensignals'
        )
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # API keys
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        
    async def initialize(self):
        """Initialize all connections"""
        try:
            # PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', 5432)),
                user=os.getenv('DB_USER', 'goldensignals'),
                password=os.getenv('DB_PASSWORD', 'password'),
                database=os.getenv('DB_NAME', 'goldensignals'),
                min_size=10,
                max_size=20
            )
            
            # SQLAlchemy async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
                pool_size=20,
                max_overflow=40
            )
            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Redis connection
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Initialize Alpha Vantage if key is available
            if self.alpha_vantage_key:
                self.alpha_vantage_ts = TimeSeries(
                    key=self.alpha_vantage_key, 
                    output_format='pandas'
                )
                self.alpha_vantage_ti = TechIndicators(
                    key=self.alpha_vantage_key,
                    output_format='pandas'
                )
            
            await self._create_tables()
            logger.info("Live data connector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize live data connector: {e}")
            raise
    
    async def _create_tables(self):
        """Create necessary database tables if they don't exist"""
        async with self.pg_pool.acquire() as conn:
            # Training data table with timezone-aware timestamp
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    open_price FLOAT NOT NULL,
                    high_price FLOAT NOT NULL,
                    low_price FLOAT NOT NULL,
                    close_price FLOAT NOT NULL,
                    volume BIGINT NOT NULL,
                    adjusted_close FLOAT,
                    features JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(symbol, timestamp)
                )
            """)
            
            # Model performance tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL,
                    model_version VARCHAR(50) NOT NULL,
                    training_date TIMESTAMPTZ NOT NULL,
                    accuracy FLOAT,
                    precision_score FLOAT,
                    recall FLOAT,
                    f1_score FLOAT,
                    sharpe_ratio FLOAT,
                    max_drawdown FLOAT,
                    total_return FLOAT,
                    parameters JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Signal history for backtesting
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_history (
                    id SERIAL PRIMARY KEY,
                    signal_id VARCHAR(36) UNIQUE NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    signal_type VARCHAR(10) NOT NULL,
                    confidence FLOAT NOT NULL,
                    entry_price FLOAT NOT NULL,
                    stop_loss FLOAT,
                    take_profit FLOAT,
                    actual_outcome VARCHAR(20),
                    profit_loss FLOAT,
                    generated_at TIMESTAMPTZ NOT NULL,
                    closed_at TIMESTAMPTZ,
                    metadata JSONB
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_data_symbol_timestamp 
                ON training_data(symbol, timestamp DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_history_symbol 
                ON signal_history(symbol, generated_at DESC)
            """)
    
    async def fetch_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch historical data from multiple sources"""
        try:
            # Ensure dates are timezone-aware
            start_date = ensure_timezone_aware(start_date)
            end_date = ensure_timezone_aware(end_date)
            
            # Try cache first
            cached_data = await self._get_cached_data(symbol, start_date, end_date)
            if cached_data is not None and len(cached_data) > 0:
                logger.info(f"Using cached data for {symbol}")
                return cached_data
            
            # Fetch from yfinance (primary source)
            logger.info(f"Fetching live data for {symbol} from {start_date} to {end_date}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty and self.alpha_vantage_key:
                # Fallback to Alpha Vantage
                logger.info(f"Falling back to Alpha Vantage for {symbol}")
                data, _ = self.alpha_vantage_ts.get_daily_adjusted(
                    symbol=symbol,
                    outputsize='full'
                )
                # Filter by date range
                data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            if not data.empty:
                # Ensure index is timezone-aware
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                
                # Store in database
                await self._store_training_data(symbol, data)
                # Cache in Redis
                await self._cache_data(symbol, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            # Try to get from database as last resort
            return await self._get_db_data(symbol, start_date, end_date)
    
    async def fetch_technical_indicators(
        self, 
        symbol: str, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate technical indicators for the data"""
        try:
            # Basic indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            
            # EMA
            data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_sma = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_upper'] = bb_sma + (bb_std * 2)
            data['BB_lower'] = bb_sma - (bb_std * 2)
            data['BB_width'] = data['BB_upper'] - data['BB_lower']
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
            
            # ATR (Average True Range)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            data['ATR'] = true_range.rolling(window=14).mean()
            
            # Store indicators in features column
            features = {
                'technical_indicators': {
                    'sma': {'20': data['SMA_20'].iloc[-1] if not data['SMA_20'].isna().all() else None},
                    'ema': {'12': data['EMA_12'].iloc[-1] if not data['EMA_12'].isna().all() else None},
                    'rsi': data['RSI'].iloc[-1] if not data['RSI'].isna().all() else None,
                    'macd': {
                        'value': data['MACD'].iloc[-1] if not data['MACD'].isna().all() else None,
                        'signal': data['MACD_signal'].iloc[-1] if not data['MACD_signal'].isna().all() else None
                    }
                }
            }
            
            # Update database with indicators
            await self._update_features(symbol, features)
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return data
    
    async def fetch_live_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch live quotes for multiple symbols"""
        quotes = {}
        
        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                quotes[symbol] = {
                    'price': info.get('regularMarketPrice', info.get('currentPrice', 0)),
                    'change': info.get('regularMarketChange', 0),
                    'changePercent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'bid': info.get('bid', 0),
                    'ask': info.get('ask', 0),
                    'high': info.get('dayHigh', 0),
                    'low': info.get('dayLow', 0),
                    'open': info.get('regularMarketOpen', 0),
                    'previousClose': info.get('regularMarketPreviousClose', 0),
                    'timestamp': now_utc().isoformat()
                }
                
                # Cache the quote if Redis is available
                if self.redis_client:
                    await self.redis_client.setex(
                        f"quote:{symbol}",
                        60,  # 1 minute TTL
                        str(quotes[symbol])
                    )
                
        except Exception as e:
            logger.error(f"Error fetching live quotes: {e}")
        
        return quotes
    
    async def prepare_training_dataset(
        self,
        symbols: List[str],
        years: int = 20,
        features: List[str] = None
    ) -> pd.DataFrame:
        """Prepare comprehensive training dataset"""
        all_data = []
        end_date = now_utc()
        start_date = end_date - timedelta(days=years * 365)
        
        for symbol in symbols:
            logger.info(f"Preparing training data for {symbol}")
            
            # Fetch historical data
            data = await self.fetch_historical_data(symbol, start_date, end_date)
            
            if not data.empty:
                # Add technical indicators
                data = await self.fetch_technical_indicators(symbol, data)
                
                # Add symbol column
                data['symbol'] = symbol
                
                # Add target variables for supervised learning
                data['target_1d'] = (data['Close'].shift(-1) > data['Close']).astype(int)
                data['target_5d'] = (data['Close'].shift(-5) > data['Close']).astype(int)
                data['returns_1d'] = data['Close'].pct_change().shift(-1)
                data['returns_5d'] = data['Close'].pct_change(5).shift(-5)
                
                all_data.append(data)
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Select features if specified
            if features:
                feature_cols = [col for col in features if col in combined_data.columns]
                combined_data = combined_data[feature_cols + ['symbol', 'target_1d', 'target_5d']]
            
            # Remove NaN values
            combined_data = combined_data.dropna()
            
            # Save to database
            await self._save_training_dataset(combined_data)
            
            logger.info(f"Prepared training dataset with {len(combined_data)} samples")
            return combined_data
        
        return pd.DataFrame()
    
    async def _store_training_data(self, symbol: str, data: pd.DataFrame):
        """Store training data in PostgreSQL"""
        async with self.pg_pool.acquire() as conn:
            for index, row in data.iterrows():
                # Ensure timestamp is timezone-aware
                timestamp = ensure_timezone_aware(index)
                
                await conn.execute("""
                    INSERT INTO training_data 
                    (symbol, timestamp, open_price, high_price, low_price, 
                     close_price, volume, adjusted_close)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, timestamp) DO UPDATE
                    SET open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        adjusted_close = EXCLUDED.adjusted_close
                """, 
                symbol, 
                timestamp,
                float(row.get('Open', 0)),
                float(row.get('High', 0)),
                float(row.get('Low', 0)),
                float(row.get('Close', 0)),
                int(row.get('Volume', 0)),
                float(row.get('Close', 0))  # Using Close as adjusted close
                )
    
    async def _cache_data(self, symbol: str, data: pd.DataFrame):
        """Cache data in Redis"""
        try:
            if not self.redis_client:
                return
                
            # Convert to JSON-serializable format
            data_dict = data.reset_index().to_dict('records')
            
            # Store with 1 hour TTL
            await self.redis_client.setex(
                f"historical:{symbol}",
                3600,
                str(data_dict)
            )
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    async def _get_cached_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Get cached data from Redis"""
        try:
            if not self.redis_client:
                return None
                
            # Ensure dates are timezone-aware
            start_date = ensure_timezone_aware(start_date)
            end_date = ensure_timezone_aware(end_date)
            
            cached = await self.redis_client.get(f"historical:{symbol}")
            if cached:
                import ast
                data_list = ast.literal_eval(cached)
                df = pd.DataFrame(data_list)
                if 'index' in df.columns:
                    df['index'] = pd.to_datetime(df['index'])
                    df.set_index('index', inplace=True)
                    # Ensure index is timezone-aware
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                
                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                return df if not df.empty else None
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
        
        return None
    
    async def _get_db_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Get data from PostgreSQL"""
        # Ensure dates are timezone-aware
        start_date = ensure_timezone_aware(start_date)
        end_date = ensure_timezone_aware(end_date)
        
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT timestamp, open_price, high_price, low_price, 
                       close_price, volume, adjusted_close
                FROM training_data
                WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
                ORDER BY timestamp
            """, symbol, start_date, end_date)
            
            if rows:
                df = pd.DataFrame(rows)
                df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                df.set_index('timestamp', inplace=True)
                return df
        
        return pd.DataFrame()
    
    async def _update_features(self, symbol: str, features: Dict):
        """Update features in database"""
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                UPDATE training_data
                SET features = $1
                WHERE symbol = $2 AND timestamp = (
                    SELECT MAX(timestamp) FROM training_data WHERE symbol = $2
                )
            """, features, symbol)
    
    async def _save_training_dataset(self, data: pd.DataFrame):
        """Save prepared training dataset"""
        # Save to parquet for efficient storage
        data.to_parquet('data/training_dataset.parquet', engine='pyarrow')
        
        # Also save metadata
        metadata = {
            'created_at': now_utc().isoformat(),
            'num_samples': len(data),
            'symbols': data['symbol'].unique().tolist(),
            'features': data.columns.tolist(),
            'date_range': {
                'start': data.index.min().isoformat() if not data.empty else None,
                'end': data.index.max().isoformat() if not data.empty else None
            }
        }
        
        if self.redis_client:
            await self.redis_client.set(
                'training_dataset_metadata',
                str(metadata)
            )
    
    async def close(self):
        """Close all connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.engine:
            await self.engine.dispose()


# Singleton instance
live_data_connector = LiveDataConnector() 