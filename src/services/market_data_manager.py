"""
Market Data Manager with Multi-Provider Support
Handles data fetching with fallback providers and circuit breakers
"""

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pa, timezonendas as pd
import yfinance as yf
import aiohttp
from cachetools import TTLCache
from functools import lru_cache

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.call_times = []
        
    async def acquire(self):
        now = datetime.now(tz=timezone.utc)
        # Remove old calls outside the period
        self.call_times = [t for t in self.call_times if now - t < timedelta(seconds=self.period)]
        
        if len(self.call_times) >= self.calls:
            # Wait until we can make a call
            wait_time = (self.call_times[0] + timedelta(seconds=self.period) - now).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                
        self.call_times.append(now)

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call_allowed(self) -> bool:
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and datetime.now(tz=timezone.utc) - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
            
    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
        
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now(tz=timezone.utc)
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def fetch_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def fetch_historical(self, symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        pass

class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider with improved error handling"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(calls=100, period=60)
        # Create a session for better performance
        self._session = None
        
    @property
    def session(self):
        if self._session is None:
            import requests_cache
            # Use cached session to reduce API calls
            self._session = requests_cache.CachedSession(
                cache_name='yfinance_cache',
                expire_after=300  # 5 minutes
            )
        return self._session
        
    async def fetch_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            await self.rate_limiter.acquire()
            loop = asyncio.get_event_loop()
            
            # Use session for better connection management
            ticker = await loop.run_in_executor(
                None, 
                lambda: yf.Ticker(symbol, session=self.session)
            )
            
            # Try fast_info first (doesn't require authentication)
            try:
                fast_info = ticker.fast_info
                return {
                    'symbol': symbol,
                    'price': float(fast_info.get('lastPrice', 0)),
                    'volume': int(fast_info.get('lastVolume', 0)),
                    'market_cap': float(fast_info.get('marketCap', 0)),
                    'timestamp': datetime.now(tz=timezone.utc),
                    'provider': 'yfinance_fast'
                }
            except:
                # Fallback to history if fast_info fails
                hist = await loop.run_in_executor(
                    None,
                    lambda: ticker.history(period="1d", interval="1m")
                )
                if not hist.empty:
                    latest = hist.iloc[-1]
                    return {
                        'symbol': symbol,
                        'price': float(latest['Close']),
                        'volume': int(latest['Volume']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'timestamp': datetime.now(tz=timezone.utc),
                        'provider': 'yfinance_history'
                    }
                    
        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {e}")
            return None
            
    async def fetch_historical(self, symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        try:
            await self.rate_limiter.acquire()
            loop = asyncio.get_event_loop()
            
            ticker = await loop.run_in_executor(
                None,
                lambda: yf.Ticker(symbol, session=self.session)
            )
            
            data = await loop.run_in_executor(
                None,
                lambda: ticker.history(period=period, interval=interval)
            )
            
            if not data.empty:
                return data
                
        except Exception as e:
            logger.error(f"YFinance historical error for {symbol}: {e}")
            
        return None

class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limiter = RateLimiter(calls=5, period=60)  # Free tier limits
        
    async def fetch_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
            
        try:
            await self.rate_limiter.acquire()
            
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'Global Quote' in data:
                            quote = data['Global Quote']
                            return {
                                'symbol': symbol,
                                'price': float(quote.get('05. price', 0)),
                                'volume': int(quote.get('06. volume', 0)),
                                'high': float(quote.get('03. high', 0)),
                                'low': float(quote.get('04. low', 0)),
                                'timestamp': datetime.now(tz=timezone.utc),
                                'provider': 'alpha_vantage'
                            }
                            
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            
        return None
        
    async def fetch_historical(self, symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        # Implement if needed
        return None

class MockDataProvider(DataProvider):
    """Mock data provider for testing and fallback"""
    
    async def fetch_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        import random
        
        # Generate realistic mock data
        base_prices = {
            'AAPL': 175.0,
            'MSFT': 380.0,
            'GOOGL': 140.0,
            'AMZN': 170.0,
            'TSLA': 250.0,
            'SPY': 450.0,
            'QQQ': 385.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        volatility = 0.02
        
        return {
            'symbol': symbol,
            'price': round(base_price * (1 + random.uniform(-volatility, volatility)), 2),
            'volume': random.randint(1000000, 10000000),
            'high': round(base_price * (1 + random.uniform(0, volatility)), 2),
            'low': round(base_price * (1 - random.uniform(0, volatility)), 2),
            'timestamp': datetime.now(tz=timezone.utc),
            'provider': 'mock'
        }
        
    async def fetch_historical(self, symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        # Generate mock historical data
        import numpy as np
        
        if interval == "1d":
            periods = 30
        elif interval == "1h":
            periods = 24 * 7
        else:
            periods = 100
            
        dates = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=periods, freq='D')
        
        # Generate realistic price movement
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, periods)
        prices = base_price * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, periods)),
            'High': prices * (1 + np.random.uniform(0, 0.02, periods)),
            'Low': prices * (1 - np.random.uniform(0, 0.02, periods)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, periods)
        }, index=dates)
        
        # Ensure logical consistency
        data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return data

class MarketDataManager:
    """Main market data manager with multi-provider support"""
    
    def __init__(self):
        # Initialize providers
        self.providers = self._initialize_providers()
        
        # Circuit breakers for each provider
        self.circuit_breakers = {
            provider.__class__.__name__: CircuitBreaker()
            for provider in self.providers
        }
        
        # Cache
        self.cache = TTLCache(maxsize=1000, ttl=300)  # 5 minute cache
        
        # Last known good data
        self.fallback_data = {}
        
    def _initialize_providers(self) -> List[DataProvider]:
        providers = []
        
        # Always add YFinance
        providers.append(YFinanceProvider())
        
        # Add Alpha Vantage if API key is available
        alpha_key = os.getenv('ALPHA_VANTAGE_KEY')
        if alpha_key:
            providers.append(AlphaVantageProvider(alpha_key))
            
        # Always add mock provider as last resort
        providers.append(MockDataProvider())
        
        logger.info(f"Initialized {len(providers)} data providers")
        return providers
        
    async def get_market_data(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get market data with fallback support"""
        
        # Check cache first
        cache_key = f"price:{symbol}"
        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit for {symbol}")
            return self.cache[cache_key]
            
        # Try each provider
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            breaker = self.circuit_breakers[provider_name]
            
            if not breaker.call_allowed():
                logger.debug(f"Circuit breaker OPEN for {provider_name}")
                continue
                
            try:
                data = await provider.fetch_price(symbol)
                if data:
                    breaker.record_success()
                    
                    # Cache the successful result
                    self.cache[cache_key] = data
                    
                    # Store as fallback
                    self.fallback_data[symbol] = data
                    
                    logger.info(f"Got data for {symbol} from {provider_name}")
                    return data
                    
            except Exception as e:
                breaker.record_failure()
                logger.error(f"Provider {provider_name} failed for {symbol}: {e}")
                
        # All providers failed, use fallback
        if symbol in self.fallback_data:
            logger.warning(f"Using fallback data for {symbol}")
            return self.fallback_data[symbol]
            
        # No data available at all
        raise Exception(f"No data available for {symbol}")
        
    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1mo", 
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get historical data with fallback support"""
        
        # Check cache
        cache_key = f"hist:{symbol}:{period}:{interval}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
            
        # Try each provider
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            breaker = self.circuit_breakers[provider_name]
            
            if not breaker.call_allowed():
                continue
                
            try:
                data = await provider.fetch_historical(symbol, period, interval)
                if data is not None and not data.empty:
                    breaker.record_success()
                    
                    # Cache the result
                    self.cache[cache_key] = data
                    
                    return data
                    
            except Exception as e:
                breaker.record_failure()
                logger.error(f"Historical data error from {provider_name}: {e}")
                
        # Generate mock data as last resort
        logger.warning(f"Using mock historical data for {symbol}")
        mock_provider = MockDataProvider()
        return await mock_provider.fetch_historical(symbol, period, interval)

# Global instance
_market_data_manager = None

def get_market_data_manager() -> MarketDataManager:
    """Get or create the global market data manager instance"""
    global _market_data_manager
    if _market_data_manager is None:
        _market_data_manager = MarketDataManager()
    return _market_data_manager 