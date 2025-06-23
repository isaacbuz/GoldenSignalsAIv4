# Live Data Solutions for GoldenSignals Trading System

## Overview

Based on extensive research into how professional trading systems handle live stock data, this guide provides solutions to common issues like rate limiting, data reliability, and real-time streaming.

## The Problem with yfinance

While yfinance is popular for prototyping, it has significant limitations for production use:

1. **Not an Official API**: yfinance scrapes Yahoo Finance endpoints without authorization
2. **Rate Limiting**: Yahoo actively blocks repeated requests (429 errors)
3. **Unreliable**: Any Yahoo site change can break functionality
4. **No Real-time Support**: Limited to delayed data and historical prices

## Professional Solutions

### 1. **Alpaca Markets** (Recommended for US Trading)
- **Real-time WebSocket streaming**
- **Free tier with paper trading**
- **Official API with proper authentication**
- **Integrated trading execution**

```python
# Example: Alpaca WebSocket for real-time data
from alpaca.data.live import StockDataStream

stream = StockDataStream(api_key, secret_key)

async def handle_bar(data):
    print(f"Bar: {data.symbol} @ ${data.close}")

stream.subscribe_bars(handle_bar, "SPY")
stream.run()
```

### 2. **Polygon.io** (High-Quality US Market Data)
- **WebSocket support for real-time streaming**
- **REST API for historical data**
- **Free tier: 5 calls/minute (limited)**
- **Paid plans for production use**

```python
from polygon import WebSocketClient

def handle_msg(msg):
    print(f"Received: {msg}")

client = WebSocketClient(api_key)
client.subscribe("T.SPY")  # Trades
client.subscribe("Q.SPY")  # Quotes
client.run(handle_msg)
```

### 3. **Finnhub** (Global Coverage)
- **60 API calls/minute (free tier)**
- **Real-time quotes with slight delay**
- **News sentiment analysis**
- **Global market coverage**

```python
import finnhub

client = finnhub.Client(api_key="YOUR_API_KEY")
quote = client.quote("AAPL")
# Returns: {'c': current, 'h': high, 'l': low, 'o': open, 'pc': prev_close}
```

### 4. **EODHD** (End-of-Day + Real-time)
- **WebSocket for real-time crypto/forex**
- **Comprehensive historical data**
- **Fundamental data included**
- **Affordable pricing**

## Implementation Strategy

### Phase 1: Enhanced WebSocket Architecture

```python
import asyncio
from typing import Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime
import websockets
import json

@dataclass
class MarketData:
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    source: str

class MultiSourceDataAggregator:
    """Aggregates data from multiple sources for reliability"""
    
    def __init__(self):
        self.sources = {}
        self.subscribers = {}
        self.data_buffer = {}
        
    async def add_source(self, name: str, connector):
        """Add a data source connector"""
        self.sources[name] = connector
        
    async def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to symbol updates"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
        
        # Start data collection from all sources
        for source_name, connector in self.sources.items():
            asyncio.create_task(
                connector.subscribe(symbol, 
                    lambda data: self._handle_data(source_name, symbol, data))
            )
    
    async def _handle_data(self, source: str, symbol: str, data: dict):
        """Handle incoming data from a source"""
        market_data = MarketData(
            symbol=symbol,
            price=data.get('price'),
            volume=data.get('volume'),
            timestamp=datetime.now(),
            source=source
        )
        
        # Notify all subscribers
        if symbol in self.subscribers:
            for callback in self.subscribers[symbol]:
                await callback(market_data)
```

### Phase 2: Rate Limit Management

```python
from requests_ratelimiter import LimiterSession, RequestRate, Limiter, Duration
import time

class RateLimitedAPIClient:
    """Base class for rate-limited API access"""
    
    def __init__(self, rate_limits: Dict[str, int]):
        """
        rate_limits: {'per_second': 1, 'per_minute': 60, 'per_hour': 360}
        """
        self.limiters = []
        
        if 'per_second' in rate_limits:
            self.limiters.append(
                RequestRate(rate_limits['per_second'], Duration.SECOND)
            )
        if 'per_minute' in rate_limits:
            self.limiters.append(
                RequestRate(rate_limits['per_minute'], Duration.MINUTE)
            )
        if 'per_hour' in rate_limits:
            self.limiters.append(
                RequestRate(rate_limits['per_hour'], Duration.HOUR)
            )
            
        limiter = Limiter(*self.limiters)
        self.session = LimiterSession(limiter=limiter)
        
        # Add retry logic
        self.session.mount('https://', 
            HTTPAdapter(max_retries=Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            ))
        )
```

### Phase 3: Data Source Connectors

```python
# Alpaca Connector
class AlpacaConnector:
    def __init__(self, api_key: str, secret_key: str):
        self.client = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockDataStream(api_key, secret_key)
        
    async def subscribe(self, symbol: str, callback):
        async def handle_bar(bar):
            await callback({
                'price': bar.close,
                'volume': bar.volume,
                'high': bar.high,
                'low': bar.low
            })
        
        self.data_client.subscribe_bars(handle_bar, symbol)
        
    async def get_historical(self, symbol: str, start: str, end: str):
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            timeframe=TimeFrame.Minute
        )
        return self.client.get_stock_bars(request)

# Polygon Connector
class PolygonConnector:
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)
        self.ws_client = WebSocketClient(api_key)
        
    async def subscribe(self, symbol: str, callback):
        def handle_msg(msgs):
            for msg in msgs:
                if msg['ev'] == 'T':  # Trade
                    asyncio.create_task(callback({
                        'price': msg['p'],
                        'volume': msg['s'],
                        'timestamp': msg['t']
                    }))
        
        await self.ws_client.subscribe(['T.' + symbol])
        self.ws_client.run_async(handle_msg)
```

### Phase 4: Fallback Strategy

```python
class DataSourceManager:
    """Manages multiple data sources with automatic fallback"""
    
    def __init__(self):
        self.primary_source = None
        self.fallback_sources = []
        self.health_check_interval = 30  # seconds
        
    async def get_quote(self, symbol: str) -> Dict:
        """Get quote with automatic fallback"""
        # Try primary source
        if self.primary_source:
            try:
                return await self._get_with_timeout(
                    self.primary_source.get_quote(symbol), 
                    timeout=5
                )
            except Exception as e:
                print(f"Primary source failed: {e}")
        
        # Try fallback sources
        for source in self.fallback_sources:
            try:
                return await self._get_with_timeout(
                    source.get_quote(symbol), 
                    timeout=5
                )
            except Exception as e:
                print(f"Fallback source {source.__class__.__name__} failed: {e}")
                continue
        
        raise Exception("All data sources failed")
    
    async def _get_with_timeout(self, coro, timeout):
        return await asyncio.wait_for(coro, timeout=timeout)
```

## Best Practices

### 1. **Use Multiple Data Sources**
- Primary: Alpaca or broker API
- Secondary: Polygon or Finnhub
- Fallback: Cached data or delayed quotes

### 2. **Implement Smart Caching**
```python
from functools import lru_cache
from datetime import datetime, timedelta

class SmartCache:
    def __init__(self, ttl_seconds=60):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache = {}
        
    def get(self, key: str):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return value
        return None
        
    def set(self, key: str, value):
        self.cache[key] = (value, datetime.now())
```

### 3. **Handle Market Hours**
```python
import pytz
from datetime import time

def is_market_open():
    """Check if US market is open"""
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # Check if weekday
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
        
    return market_open <= now.time() <= market_close
```

### 4. **Error Handling and Monitoring**
```python
import logging
from prometheus_client import Counter, Histogram

# Metrics
api_calls = Counter('api_calls_total', 'Total API calls', ['source', 'status'])
api_latency = Histogram('api_latency_seconds', 'API call latency', ['source'])

class MonitoredAPIClient:
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.logger = logging.getLogger(source_name)
        
    async def make_request(self, endpoint: str, **kwargs):
        start_time = time.time()
        try:
            response = await self._request(endpoint, **kwargs)
            api_calls.labels(source=self.source_name, status='success').inc()
            return response
        except Exception as e:
            api_calls.labels(source=self.source_name, status='error').inc()
            self.logger.error(f"API request failed: {e}")
            raise
        finally:
            api_latency.labels(source=self.source_name).observe(
                time.time() - start_time
            )
```

## Migration Path from yfinance

### Step 1: Gradual Migration
```python
class HybridDataProvider:
    """Use yfinance for historical, real APIs for live data"""
    
    def __init__(self, live_provider):
        self.live_provider = live_provider
        
    async def get_data(self, symbol: str, period: str = "1d"):
        if period in ["1d", "5d", "1mo"]:  # Historical
            # Use yfinance with rate limiting
            session = LimiterSession(per_second=0.5)
            return yf.download(symbol, period=period, session=session)
        else:  # Real-time
            return await self.live_provider.get_quote(symbol)
```

### Step 2: Data Validation
```python
def validate_market_data(data: Dict) -> bool:
    """Validate incoming market data"""
    required_fields = ['price', 'volume', 'timestamp']
    
    # Check required fields
    if not all(field in data for field in required_fields):
        return False
    
    # Validate price
    if not 0 < data['price'] < 1000000:
        return False
    
    # Validate timestamp
    if abs((datetime.now() - data['timestamp']).seconds) > 300:  # 5 min old
        return False
    
    return True
```

## Cost Comparison

| Provider | Free Tier | Paid Starting | Best For |
|----------|-----------|---------------|----------|
| Alpaca | Full (paper trading) | $0 (live trading) | US stocks, trading |
| Polygon | 5 calls/min | $29/month | High-quality US data |
| Finnhub | 60 calls/min | $50/month | Global coverage |
| EODHD | 20 calls/day | $20/month | Historical + fundamentals |
| IEX Cloud | 50k calls/month | $19/month | US stocks |

## Conclusion

For GoldenSignals, the recommended approach is:

1. **Primary**: Alpaca API (free, reliable, WebSocket support)
2. **Secondary**: Finnhub (good free tier, global coverage)
3. **Historical**: Continue using yfinance with strict rate limiting
4. **Architecture**: Multi-source aggregator with automatic fallback

This provides a robust, scalable solution that can handle production workloads while maintaining cost efficiency. 