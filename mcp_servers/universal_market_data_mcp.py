"""
Universal Market Data MCP Server
Provides standardized access to all market data sources
Issue #190: MCP-1: Build Universal Market Data MCP Server
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Union
import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
import yfinance as yf
import pandas as pd
import numpy as np
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources"""
    YAHOO = "yahoo"
    MOCK = "mock"
    IB = "interactive_brokers"
    BLOOMBERG = "bloomberg"


class AssetClass(Enum):
    """Asset classes supported"""
    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"


class MarketDataCache:
    """Simple in-memory cache for market data"""
    
    def __init__(self, ttl_seconds: int = 60):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
        
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: Dict[str, Any]):
        """Cache data with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()


class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    async def check_limit(self, source: str) -> bool:
        """Check if request is within rate limit"""
        now = time.time()
        
        # Clean old requests
        self.requests[source] = [
            req_time for req_time in self.requests[source]
            if now - req_time < self.window
        ]
        
        # Check limit
        if len(self.requests[source]) >= self.max_requests:
            return False
            
        # Record request
        self.requests[source].append(now)
        return True


class UniversalMarketDataMCP:
    """
    MCP Server providing standardized access to all market data
    Handles multiple data sources, caching, rate limiting, and failover
    """
    
    def __init__(self):
        self.app = FastAPI(title="Universal Market Data MCP Server")
        self.cache = MarketDataCache(ttl_seconds=60)
        self.rate_limiter = RateLimiter()
        self.websocket_clients: List[WebSocket] = []
        self.data_sources = {
            DataSource.YAHOO: self._yahoo_adapter,
            DataSource.MOCK: self._mock_adapter,
        }
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            return {"service": "Universal Market Data MCP", "status": "active"}
        
        @self.app.get("/tools")
        async def list_tools():
            """List available tools and their parameters"""
            return {
                "tools": [
                    {
                        "name": "get_price",
                        "description": "Get current price for symbol",
                        "parameters": {
                            "symbol": "string (required)",
                            "asset_class": "string (optional, default: equity)"
                        }
                    },
                    {
                        "name": "get_orderbook",
                        "description": "Get order book data",
                        "parameters": {
                            "symbol": "string (required)",
                            "depth": "integer (optional, default: 10)"
                        }
                    },
                    {
                        "name": "get_historical",
                        "description": "Get historical price data",
                        "parameters": {
                            "symbol": "string (required)",
                            "start_date": "string (YYYY-MM-DD)",
                            "end_date": "string (YYYY-MM-DD)",
                            "interval": "string (1m, 5m, 1h, 1d)"
                        }
                    },
                    {
                        "name": "get_quote",
                        "description": "Get detailed quote information",
                        "parameters": {
                            "symbol": "string (required)"
                        }
                    },
                    {
                        "name": "get_market_status",
                        "description": "Get market open/close status",
                        "parameters": {}
                    }
                ]
            }
        
        @self.app.post("/call")
        async def call_tool(request: Dict[str, Any]):
            """Execute a tool call"""
            tool_name = request.get("tool")
            params = request.get("parameters", {})
            
            try:
                if tool_name == "get_price":
                    return await self._get_price(params)
                elif tool_name == "get_orderbook":
                    return await self._get_orderbook(params)
                elif tool_name == "get_historical":
                    return await self._get_historical(params)
                elif tool_name == "get_quote":
                    return await self._get_quote(params)
                elif tool_name == "get_market_status":
                    return await self._get_market_status()
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
                    
            except Exception as e:
                logger.error(f"Error in tool call {tool_name}: {e}")
                return {"error": str(e), "tool": tool_name}
        
        @self.app.websocket("/stream")
        async def websocket_stream(websocket: WebSocket):
            """WebSocket endpoint for real-time data streaming"""
            await websocket.accept()
            self.websocket_clients.append(websocket)
            
            try:
                while True:
                    data = await websocket.receive_text()
                    request = json.loads(data)
                    
                    if request.get('action') == 'subscribe':
                        await self._handle_subscription(websocket, request.get('symbols', []))
                    elif request.get('action') == 'unsubscribe':
                        await self._handle_unsubscription(websocket, request.get('symbols', []))
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_clients.remove(websocket)
                await websocket.close()
    
    async def _get_price(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get current price for a symbol"""
        symbol = params.get('symbol')
        if not symbol:
            raise ValueError("Symbol is required")
            
        asset_class = params.get('asset_class', 'equity')
        
        # Check cache first
        cache_key = f"price:{symbol}:{asset_class}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Select data source
        source = self._select_data_source(asset_class)
        
        # Check rate limit
        if not await self.rate_limiter.check_limit(source.value):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Fetch data
        try:
            adapter = self.data_sources[source]
            price_data = await adapter('price', symbol, asset_class)
            
            result = {
                'symbol': symbol,
                'price': price_data['price'],
                'bid': price_data.get('bid'),
                'ask': price_data.get('ask'),
                'volume': price_data.get('volume'),
                'change': price_data.get('change', 0),
                'change_percent': price_data.get('change_percent', 0),
                'timestamp': datetime.now().isoformat(),
                'source': source.value
            }
            
            # Cache result
            self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            # Try failover
            logger.warning(f"Primary source failed: {e}")
            return await self._failover_get_price(symbol, asset_class, str(e))
    
    async def _get_orderbook(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get order book data"""
        symbol = params.get('symbol')
        depth = params.get('depth', 10)
        
        if not symbol:
            raise ValueError("Symbol is required")
        
        # For now, return mock order book data
        # In production, this would connect to real order book feeds
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'bids': [
                {'price': 100.00 - i * 0.01, 'size': 100 + i * 10}
                for i in range(depth)
            ],
            'asks': [
                {'price': 100.01 + i * 0.01, 'size': 100 + i * 10}
                for i in range(depth)
            ],
            'spread': 0.01,
            'mid_price': 100.005
        }
    
    async def _get_historical(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical price data"""
        symbol = params.get('symbol')
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        interval = params.get('interval', '1d')
        
        if not all([symbol, start_date, end_date]):
            raise ValueError("Symbol, start_date, and end_date are required")
        
        # Check cache
        cache_key = f"historical:{symbol}:{start_date}:{end_date}:{interval}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # Use Yahoo Finance for historical data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Convert to list of dictionaries
            data = []
            for idx, row in df.iterrows():
                data.append({
                    'date': idx.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            result = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'interval': interval,
                'data_points': len(data),
                'data': data
            }
            
            # Cache result
            self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_quote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed quote information"""
        symbol = params.get('symbol')
        if not symbol:
            raise ValueError("Symbol is required")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'exchange': info.get('exchange'),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_market_status(self) -> Dict[str, Any]:
        """Get market open/close status"""
        now = datetime.now()
        
        # Simple US market hours check (9:30 AM - 4:00 PM ET)
        # In production, this would be more sophisticated
        market_open = False
        if now.weekday() < 5:  # Monday-Friday
            # Convert to ET (simplified)
            hour = now.hour
            if 9 <= hour < 16:
                market_open = True
        
        return {
            'timestamp': now.isoformat(),
            'us_equity_market': 'open' if market_open else 'closed',
            'forex_market': 'open',  # 24/5
            'crypto_market': 'open',  # 24/7
            'next_open': self._get_next_market_open(),
            'next_close': self._get_next_market_close()
        }
    
    def _select_data_source(self, asset_class: str) -> DataSource:
        """Select appropriate data source based on asset class"""
        # Simplified logic - in production would be more sophisticated
        if asset_class == 'equity':
            return DataSource.YAHOO
        else:
            return DataSource.MOCK
    
    async def _yahoo_adapter(self, data_type: str, symbol: str, asset_class: str) -> Dict[str, Any]:
        """Adapter for Yahoo Finance data"""
        if data_type == 'price':
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'price': info.get('regularMarketPrice', info.get('previousClose', 0)),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'volume': info.get('volume'),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0)
            }
        
        raise ValueError(f"Unsupported data type: {data_type}")
    
    async def _mock_adapter(self, data_type: str, symbol: str, asset_class: str) -> Dict[str, Any]:
        """Mock data adapter for testing"""
        if data_type == 'price':
            # Generate random price data
            base_price = 100.0
            spread = 0.01
            
            return {
                'price': base_price + np.random.randn() * 2,
                'bid': base_price - spread/2,
                'ask': base_price + spread/2,
                'volume': int(np.random.uniform(1000000, 10000000)),
                'change': np.random.randn() * 0.5,
                'change_percent': np.random.randn() * 2
            }
        
        raise ValueError(f"Unsupported data type: {data_type}")
    
    async def _failover_get_price(self, symbol: str, asset_class: str, error: str) -> Dict[str, Any]:
        """Failover mechanism for getting price"""
        # Try mock data as failover
        try:
            price_data = await self._mock_adapter('price', symbol, asset_class)
            
            return {
                'symbol': symbol,
                'price': price_data['price'],
                'bid': price_data.get('bid'),
                'ask': price_data.get('ask'),
                'volume': price_data.get('volume'),
                'timestamp': datetime.now().isoformat(),
                'source': 'mock_failover',
                'original_error': error
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"All data sources failed: {error}, {e}")
    
    async def _handle_subscription(self, websocket: WebSocket, symbols: List[str]):
        """Handle WebSocket subscription"""
        # Start streaming data for requested symbols
        while True:
            try:
                data = {}
                for symbol in symbols:
                    price_data = await self._get_price({'symbol': symbol})
                    data[symbol] = price_data
                
                await websocket.send_json({
                    'type': 'price_update',
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Stream every second
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                break
    
    async def _handle_unsubscription(self, websocket: WebSocket, symbols: List[str]):
        """Handle WebSocket unsubscription"""
        # In a real implementation, would stop streaming for these symbols
        await websocket.send_json({
            'type': 'unsubscribed',
            'symbols': symbols,
            'timestamp': datetime.now().isoformat()
        })
    
    def _get_next_market_open(self) -> str:
        """Get next market open time"""
        now = datetime.now()
        
        # Find next weekday at 9:30 AM
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # If past 9:30 AM today, go to next day
        if now.hour >= 9 and now.minute >= 30:
            next_open += timedelta(days=1)
        
        # Skip to Monday if weekend
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        
        return next_open.isoformat()
    
    def _get_next_market_close(self) -> str:
        """Get next market close time"""
        now = datetime.now()
        
        # Find next close at 4:00 PM
        next_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # If past 4:00 PM today, go to next day
        if now.hour >= 16:
            next_close += timedelta(days=1)
        
        # Skip to Monday if weekend
        while next_close.weekday() >= 5:
            next_close += timedelta(days=1)
        
        return next_close.isoformat()


# Demo and testing
async def demo_market_data_mcp():
    """Demonstrate the Universal Market Data MCP Server"""
    
    # Create mock MCP server
    mcp = UniversalMarketDataMCP()
    
    print("Universal Market Data MCP Demo")
    print("="*50)
    
    # Test get_price
    print("\n1. Testing get_price:")
    price_result = await mcp._get_price({'symbol': 'AAPL'})
    print(f"AAPL Price: ${price_result['price']:.2f}")
    print(f"Source: {price_result['source']}")
    
    # Test get_orderbook
    print("\n2. Testing get_orderbook:")
    orderbook = await mcp._get_orderbook({'symbol': 'AAPL', 'depth': 5})
    print(f"Best Bid: ${orderbook['bids'][0]['price']:.2f} x {orderbook['bids'][0]['size']}")
    print(f"Best Ask: ${orderbook['asks'][0]['price']:.2f} x {orderbook['asks'][0]['size']}")
    print(f"Spread: ${orderbook['spread']:.2f}")
    
    # Test market status
    print("\n3. Testing market status:")
    status = await mcp._get_market_status()
    print(f"US Equity Market: {status['us_equity_market']}")
    print(f"Crypto Market: {status['crypto_market']}")
    
    # Test caching
    print("\n4. Testing cache:")
    start = time.time()
    await mcp._get_price({'symbol': 'AAPL'})
    first_call = time.time() - start
    
    start = time.time()
    await mcp._get_price({'symbol': 'AAPL'})
    cached_call = time.time() - start
    
    print(f"First call: {first_call*1000:.2f}ms")
    print(f"Cached call: {cached_call*1000:.2f}ms")
    print(f"Speed improvement: {first_call/cached_call:.1f}x")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_market_data_mcp())
    
    # To run as a server:
    # import uvicorn
    # mcp = UniversalMarketDataMCP()
    # uvicorn.run(mcp.app, host="0.0.0.0", port=8000) 