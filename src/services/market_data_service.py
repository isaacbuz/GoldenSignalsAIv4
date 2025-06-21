#!/usr/bin/env python3
"""
🚀 GoldenSignalsAI V3 - Live Market Data Service
Real-time market data streaming and processing for institutional trading

Features:
- Real-time price feeds via Yahoo Finance
- WebSocket streaming for frontend
- Technical indicator calculation
- Signal generation pipeline
- Risk assessment integration
- Performance monitoring
- After-hours data handling
- Smart error detection
"""

import asyncio
import json
import websockets
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from dataclasses import dataclass, asdict
import threading
import time as time_module
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
import pytz
from enum import Enum
from cachetools import TTLCache
import aiohttp

from src.core.database import DatabaseManager
from src.core.redis_manager import RedisManager
from src.models.market_data import MarketData
from src.services.rate_limit_handler import get_rate_limit_handler, RequestPriority
from src.services.websocket_service import get_websocket_service, MarketUpdate
from src.services.cache_service import get_cache_service, DataType
from src.services.monitoring_service import get_monitoring_service
# from src.utils.cache import cache_decorator
# from src.utils.rate_limiter import RateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataUnavailableReason(Enum):
    """Reasons why market data might be unavailable"""
    MARKET_CLOSED = "market_closed"
    INVALID_SYMBOL = "invalid_symbol"
    NETWORK_ERROR = "network_error"
    API_LIMIT = "api_limit"
    NO_DATA = "no_data"
    UNKNOWN = "unknown"

@dataclass
class MarketHours:
    """Market hours information"""
    is_open: bool
    current_time: datetime
    market_open: time
    market_close: time
    next_open: datetime
    reason: str
    timezone: str = "US/Eastern"

@dataclass 
class MarketDataError:
    """Market data error information"""
    symbol: str
    reason: DataUnavailableReason
    message: str
    is_recoverable: bool
    suggested_action: str
    timestamp: str

@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    spread: float
    timestamp: str
    change: float
    change_percent: float
    
@dataclass
class SignalData:
    """Trading signal data"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    price_target: float
    stop_loss: float
    risk_score: float
    indicators: Dict[str, float]
    timestamp: str
    is_after_hours: bool = False  # New field

class TechnicalIndicators:
    """Real-time technical indicator calculations"""
    
    @staticmethod
    def calculate_indicators(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators with NaN handling"""
        if len(data) < 10:  # Reduced from 50 to 10
            return {}
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        def safe_float(value, default=0.0):
            """Convert to float, handling NaN values"""
            try:
                if pd.isna(value) or np.isnan(value):
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default
        
        # Moving averages
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        ema_12 = close.ewm(span=12).mean().iloc[-1]
        ema_26 = close.ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        macd_line = ema_12 - ema_26
        signal_line = pd.Series(macd_line).ewm(span=9).mean().iloc[-1]
        macd_histogram = macd_line - signal_line
        
        # Bollinger Bands
        bb_middle = sma_20
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
        
        # Volatility
        volatility = close.pct_change().rolling(20).std().iloc[-1]
        
        # Volume indicators
        volume_sma = volume.rolling(20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / volume_sma if volume_sma > 0 else 1
        
        return {
            'sma_20': safe_float(sma_20),
            'sma_50': safe_float(sma_50),
            'ema_12': safe_float(ema_12),
            'ema_26': safe_float(ema_26),
            'rsi': safe_float(rsi),
            'macd': safe_float(macd_line),
            'macd_signal': safe_float(signal_line),
            'macd_histogram': safe_float(macd_histogram),
            'bb_upper': safe_float(bb_upper),
            'bb_middle': safe_float(bb_middle),
            'bb_lower': safe_float(bb_lower),
            'bb_width': safe_float(bb_width),
            'volatility': safe_float(volatility),
            'volume_ratio': safe_float(volume_ratio),
            'current_price': safe_float(close.iloc[-1])
        }

class MLModelLoader:
    """Load and use trained ML models"""
    
    def __init__(self, model_dir: str = None):
        # Auto-detect correct path based on current working directory
        if model_dir is None:
            import os
            if os.path.exists("../../ml_training/models"):
                model_dir = "../../ml_training/models"
            elif os.path.exists("../ml_training/models"):
                model_dir = "../ml_training/models"
            elif os.path.exists("ml_training/models"):
                model_dir = "ml_training/models"
            else:
                model_dir = "../../ml_training/models"  # fallback
        
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load forecast model
            with open(f"{self.model_dir}/forecast_model.pkl", 'rb') as f:
                self.models['forecast'] = pickle.load(f)
            with open(f"{self.model_dir}/forecast_scaler.pkl", 'rb') as f:
                self.scalers['forecast'] = pickle.load(f)
            
            # Load signal classifier
            with open(f"{self.model_dir}/signal_classifier.pkl", 'rb') as f:
                self.models['signal'] = pickle.load(f)
            with open(f"{self.model_dir}/signal_classifier_scaler.pkl", 'rb') as f:
                self.scalers['signal'] = pickle.load(f)
            
            # Load risk model
            with open(f"{self.model_dir}/risk_model.pkl", 'rb') as f:
                self.models['risk'] = pickle.load(f)
            with open(f"{self.model_dir}/risk_scaler.pkl", 'rb') as f:
                self.scalers['risk'] = pickle.load(f)
            
            logger.info("✅ ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self.models = {}
            self.scalers = {}
    
    def predict_price_movement(self, features: np.ndarray) -> float:
        """Predict future price movement"""
        if 'forecast' not in self.models:
            return 0.0
        
        try:
            scaled_features = self.scalers['forecast'].transform(features.reshape(1, -1))
            prediction = self.models['forecast'].predict(scaled_features)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0
    
    def classify_signal(self, features: np.ndarray) -> Dict[str, float]:
        """Classify trading signal"""
        if 'signal' not in self.models:
            return {'neutral': 1.0, 'bull': 0.0, 'bear': 0.0}
        
        try:
            scaled_features = self.scalers['signal'].transform(features.reshape(1, -1))
            proba = self.models['signal'].predict_proba(scaled_features)[0]
            
            # Map probabilities to signal types
            return {
                'neutral': float(proba[0]),
                'bull': float(proba[1]) if len(proba) > 1 else 0.0,
                'bear': float(proba[2]) if len(proba) > 2 else 0.0
            }
        except Exception as e:
            logger.error(f"Signal classification error: {e}")
            return {'neutral': 1.0, 'bull': 0.0, 'bear': 0.0}
    
    def assess_risk(self, features: np.ndarray) -> float:
        """Assess risk score"""
        if 'risk' not in self.models:
            return 0.5
        
        try:
            scaled_features = self.scalers['risk'].transform(features.reshape(1, -1))
            risk_score = self.models['risk'].predict(scaled_features)[0]
            return float(np.clip(risk_score, 0, 1))
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return 0.5

class MarketDataCache:
    """Cache for market data during off-hours"""
    
    def __init__(self, cache_dir: str = "data/market_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache: Dict[str, Dict] = {}
        self.cache_ttl = 3600 * 24  # 24 hours
        
    def save_tick(self, symbol: str, tick: MarketTick):
        """Save market tick to cache"""
        self.memory_cache[f"tick_{symbol}"] = {
            "data": asdict(tick),
            "timestamp": datetime.now().timestamp()
        }
        
        # Also persist to disk
        try:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_tick.json")
            with open(cache_file, 'w') as f:
                json.dump(asdict(tick), f)
        except Exception as e:
            logger.error(f"Failed to persist tick cache for {symbol}: {e}")
    
    def get_tick(self, symbol: str) -> Optional[MarketTick]:
        """Get cached tick if available"""
        # Try memory cache first
        cache_key = f"tick_{symbol}"
        if cache_key in self.memory_cache:
            cached = self.memory_cache[cache_key]
            age = datetime.now().timestamp() - cached["timestamp"]
            if age < self.cache_ttl:
                data = cached["data"]
                return MarketTick(**data)
        
        # Try disk cache
        try:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_tick.json")
            if os.path.exists(cache_file):
                mtime = os.path.getmtime(cache_file)
                age = datetime.now().timestamp() - mtime
                if age < self.cache_ttl:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    return MarketTick(**data)
        except Exception as e:
            logger.error(f"Failed to load tick cache for {symbol}: {e}")
        
        return None
    
    def save_historical(self, symbol: str, data: pd.DataFrame):
        """Save historical data to cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_historical.pkl")
            data.to_pickle(cache_file)
            self.memory_cache[f"hist_{symbol}"] = {
                "timestamp": datetime.now().timestamp()
            }
        except Exception as e:
            logger.error(f"Failed to save historical cache for {symbol}: {e}")
    
    def get_historical(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached historical data if available"""
        cache_key = f"hist_{symbol}"
        cache_file = os.path.join(self.cache_dir, f"{symbol}_historical.pkl")
        
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            age = datetime.now().timestamp() - mtime
            if age < self.cache_ttl:
                try:
                    return pd.read_pickle(cache_file)
                except Exception as e:
                    logger.error(f"Failed to load historical cache for {symbol}: {e}")
        
        return None

class MarketDataService:
    """Service for fetching live market data from various sources"""
    
    def __init__(self):
        # Initialize all services
        self.rate_limit_handler = get_rate_limit_handler()
        self.websocket_service = get_websocket_service()
        self.cache_service = get_cache_service()
        self.monitoring_service = get_monitoring_service()
        
        # Cache for market data (5 minute TTL)
        self.quote_cache = TTLCache(maxsize=1000, ttl=300)
        self.historical_cache = TTLCache(maxsize=500, ttl=600)
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 0.1  # 100ms between requests
        
        # WebSocket will be initialized when needed
        self._websocket_initialized = False
    
    async def _ensure_websocket(self):
        """Ensure WebSocket is initialized"""
        if not self._websocket_initialized:
            self._websocket_initialized = True
            await self.websocket_service.connect()
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a symbol using all services
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Quote data dictionary or None
        """
        # Ensure WebSocket is initialized
        await self._ensure_websocket()
        
        start_time = time.time()
        
        try:
            # Try cache first
            quote_data = await self.cache_service.get(
                DataType.QUOTE,
                symbol,
                fetch_func=lambda: self._fetch_quote(symbol)
            )
            
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            self.monitoring_service.record_latency("get_quote", latency_ms, {"symbol": symbol})
            
            if quote_data:
                self.monitoring_service.record_cache_hit("total", True)
                logger.debug(f"Fetched quote for {symbol}: ${quote_data.get('price', 0)}")
            else:
                self.monitoring_service.record_cache_hit("total", False)
            
            return quote_data
            
        except Exception as e:
            self.monitoring_service.record_error("quote_fetch", {"symbol": symbol})
            logger.error(f"Failed to fetch quote for {symbol}: {str(e)}")
            return None
    
    async def _fetch_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch quote from rate limit handler"""
        self.monitoring_service.record_api_call("quote", "rate_limit_handler")
        
        quote_data = await self.rate_limit_handler.get_quote(
            symbol, 
            priority=RequestPriority.NORMAL
        )
        
        return quote_data
    
    async def subscribe_to_symbol(self, symbol: str, callback: Callable):
        """Subscribe to real-time updates for a symbol"""
        # Ensure WebSocket is initialized
        await self._ensure_websocket()
        
        # Subscribe via WebSocket
        sub_id = await self.websocket_service.subscribe(
            [symbol],
            callback=callback
        )
        
        logger.info(f"Subscribed to {symbol} with ID: {sub_id}")
        return sub_id
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for multiple symbols using batch processing
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary of symbol -> quote data
        """
        start_time = time.time()
        
        try:
            # Use rate limit handler's batch processing
            quotes = await self.rate_limit_handler.batch_get_quotes(
                symbols,
                priority=RequestPriority.NORMAL
            )
            
            # Cache results
            for symbol, quote in quotes.items():
                if quote:
                    await self.cache_service.set(DataType.QUOTE, symbol, quote)
            
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            self.monitoring_service.record_latency("batch_quotes", latency_ms, {"count": str(len(symbols))})
            self.monitoring_service.record_api_call("batch_quotes", "rate_limit_handler")
            
            return quotes
            
        except Exception as e:
            self.monitoring_service.record_error("batch_quotes", {"count": str(len(symbols))})
            logger.error(f"Failed to fetch multiple quotes: {str(e)}")
            return {}
    
    async def get_historical_data(
        self,
        symbol: str,
        period: str = "1d",
        interval: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a symbol using caching
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical data or None
        """
        start_time = time.time()
        cache_key = f"{symbol}_{period}_{interval}"
        
        try:
            # Try cache first
            hist_data = await self.cache_service.get(
                DataType.HISTORICAL,
                cache_key,
                fetch_func=lambda: self._fetch_historical(symbol, period, interval)
            )
            
            if hist_data is not None and not hist_data.empty:
                # Add technical indicators
                hist_data = self._add_technical_indicators(hist_data)
                
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                self.monitoring_service.record_latency("historical_data", latency_ms, {
                    "symbol": symbol,
                    "period": period,
                    "interval": interval
                })
                
                logger.debug(f"Fetched {len(hist_data)} historical records for {symbol}")
            
            return hist_data
            
        except Exception as e:
            self.monitoring_service.record_error("historical_data", {"symbol": symbol})
            logger.error(f"Failed to fetch historical data for {symbol}: {str(e)}")
            return None
    
    async def _fetch_historical(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from rate limit handler"""
        self.monitoring_service.record_api_call("historical", "rate_limit_handler")
        
        hist_data = await self.rate_limit_handler.get_historical_data(
            symbol,
            period=period,
            interval=interval,
            priority=RequestPriority.NORMAL
        )
        
        return hist_data
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return self.monitoring_service.get_dashboard_data()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache_service.get_stats()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the dataframe"""
        try:
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to add technical indicators: {str(e)}")
            return df
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            # Check if market is open
            now = datetime.now()
            
            # Simple market hours check (NYSE)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
            is_market_hours = market_open <= now <= market_close
            
            is_open = is_weekday and is_market_hours
            
            # Get major indices
            indices = await self.get_quotes(['SPY', 'QQQ', 'DIA', 'IWM'])
            
            return {
                "is_open": is_open,
                "current_time": now.isoformat(),
                "market_open": market_open.isoformat(),
                "market_close": market_close.isoformat(),
                "indices": indices,
                "status": "open" if is_open else "closed"
            }
            
        except Exception as e:
            logger.error(f"Failed to get market status: {str(e)}")
            return {
                "is_open": False,
                "status": "unknown",
                "error": str(e)
            }
    
    async def search_symbols(self, query: str) -> List[Dict[str, str]]:
        """Search for symbols matching the query"""
        try:
            # Use yfinance search functionality
            # Note: This is a simplified implementation
            # In production, you might want to use a proper symbol search API
            
            # Common symbols for demo
            all_symbols = [
                {"symbol": "AAPL", "name": "Apple Inc."},
                {"symbol": "GOOGL", "name": "Alphabet Inc."},
                {"symbol": "MSFT", "name": "Microsoft Corporation"},
                {"symbol": "AMZN", "name": "Amazon.com Inc."},
                {"symbol": "TSLA", "name": "Tesla Inc."},
                {"symbol": "META", "name": "Meta Platforms Inc."},
                {"symbol": "NVDA", "name": "NVIDIA Corporation"},
                {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
                {"symbol": "V", "name": "Visa Inc."},
                {"symbol": "JNJ", "name": "Johnson & Johnson"},
                {"symbol": "WMT", "name": "Walmart Inc."},
                {"symbol": "PG", "name": "Procter & Gamble Co."},
                {"symbol": "MA", "name": "Mastercard Inc."},
                {"symbol": "UNH", "name": "UnitedHealth Group Inc."},
                {"symbol": "HD", "name": "The Home Depot Inc."},
                {"symbol": "DIS", "name": "The Walt Disney Company"},
                {"symbol": "BAC", "name": "Bank of America Corp."},
                {"symbol": "ADBE", "name": "Adobe Inc."},
                {"symbol": "CRM", "name": "Salesforce Inc."},
                {"symbol": "NFLX", "name": "Netflix Inc."},
                {"symbol": "SPY", "name": "SPDR S&P 500 ETF"},
                {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
                {"symbol": "IWM", "name": "iShares Russell 2000 ETF"},
                {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Average ETF"}
            ]
            
            # Filter based on query
            query_lower = query.lower()
            results = [
                s for s in all_symbols
                if query_lower in s["symbol"].lower() or query_lower in s["name"].lower()
            ]
            
            return results[:10]  # Limit to 10 results
            
        except Exception as e:
            logger.error(f"Failed to search symbols: {str(e)}")
            return []
    
    async def get_options_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get options chain data for a symbol"""
        try:
            # Rate limiting
            await self._rate_limit(symbol)
            
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                return None
            
            # Get options for the nearest expiration
            nearest_expiry = expirations[0]
            
            # Get calls and puts
            opt = ticker.option_chain(nearest_expiry)
            calls = opt.calls.to_dict('records')
            puts = opt.puts.to_dict('records')
            
            return {
                "symbol": symbol,
                "expirations": list(expirations),
                "selected_expiry": nearest_expiry,
                "calls": calls[:20],  # Limit to 20 strikes
                "puts": puts[:20],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch options chain for {symbol}: {str(e)}")
            return None
    
    async def get_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest news for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            # Format news items
            formatted_news = []
            for item in news[:limit]:
                formatted_news.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat(),
                    "type": item.get("type", ""),
                    "thumbnail": item.get("thumbnail", {}).get("resolutions", [{}])[0].get("url", "") if item.get("thumbnail") else ""
                })
            
            return formatted_news
            
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {str(e)}")
            return []

def main():
    """Main function for testing"""
    print("🔥 GoldenSignalsAI V3 - Market Data Service")
    print("=" * 50)
    
    service = MarketDataService()
    
    # Test market summary
    summary = service.get_market_status()
    print(f"📊 Market Summary: {summary['status']}")
    
    # Start service
    try:
        service.start_service()
    except KeyboardInterrupt:
        print("\n👋 Service stopped")

if __name__ == "__main__":
    main() 