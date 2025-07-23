"""
Live Data Fetcher for GoldenSignalsAI V2
Connects to multiple data sources and provides unified data feed for agents
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd
import requests

# Data source imports
import yfinance as yf

import websocket

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    timestamp: datetime
    price: float
    bid: float
    ask: float
    volume: int
    high: float
    low: float
    open: float
    close: float
    vwap: float = 0.0
    trades_count: int = 0
    
@dataclass
class OptionsData:
    """Standardized options data structure"""
    symbol: str
    strike: float
    expiration: str
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

class DataSourceBase:
    """Base class for all data sources"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self.last_update = None
        
    async def connect(self):
        """Connect to data source"""
        raise NotImplementedError
        
    async def disconnect(self):
        """Disconnect from data source"""
        raise NotImplementedError
        
    async def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get current quote for symbol"""
        raise NotImplementedError
        
    async def get_options_chain(self, symbol: str) -> List[OptionsData]:
        """Get options chain for symbol"""
        raise NotImplementedError

class YahooFinanceSource(DataSourceBase):
    """Yahoo Finance data source (free)"""
    
    def __init__(self):
        super().__init__("YahooFinance")
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def connect(self):
        """No connection needed for Yahoo Finance"""
        self.is_connected = True
        logger.info(f"✅ {self.name} connected")
        
    async def disconnect(self):
        """Clean up executor"""
        self.executor.shutdown(wait=True)
        self.is_connected = False
        
    def _fetch_quote_sync(self, symbol: str) -> Optional[Dict]:
        """Synchronous quote fetch for thread executor"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get latest price data
            hist = ticker.history(period="1d", interval="1m", prepost=True)
            if hist.empty:
                return None
                
            latest = hist.iloc[-1]
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': info.get('regularMarketPrice', latest['Close']),
                'bid': info.get('bid', latest['Close'] * 0.999),
                'ask': info.get('ask', latest['Close'] * 1.001),
                'volume': int(info.get('volume', latest['Volume'])),
                'high': info.get('dayHigh', latest['High']),
                'low': info.get('dayLow', latest['Low']),
                'open': info.get('regularMarketOpen', latest['Open']),
                'close': latest['Close'],
                'vwap': latest['Close']  # Approximation
            }
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None
            
    async def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get quote asynchronously"""
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(self.executor, self._fetch_quote_sync, symbol)
        
        if data:
            return MarketData(**data)
        return None
        
    def _fetch_options_sync(self, symbol: str) -> List[Dict]:
        """Synchronous options fetch"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return []
                
            options_list = []
            
            # Get first 3 expirations
            for exp in expirations[:3]:
                opt_chain = ticker.option_chain(exp)
                
                # Process calls
                for _, row in opt_chain.calls.iterrows():
                    options_list.append({
                        'symbol': symbol,
                        'strike': float(row['strike']),
                        'expiration': exp,
                        'option_type': 'call',
                        'bid': float(row.get('bid', 0)),
                        'ask': float(row.get('ask', 0)),
                        'last': float(row.get('lastPrice', 0)),
                        'volume': int(row.get('volume', 0)),
                        'open_interest': int(row.get('openInterest', 0)),
                        'implied_volatility': float(row.get('impliedVolatility', 0))
                    })
                    
                # Process puts
                for _, row in opt_chain.puts.iterrows():
                    options_list.append({
                        'symbol': symbol,
                        'strike': float(row['strike']),
                        'expiration': exp,
                        'option_type': 'put',
                        'bid': float(row.get('bid', 0)),
                        'ask': float(row.get('ask', 0)),
                        'last': float(row.get('lastPrice', 0)),
                        'volume': int(row.get('volume', 0)),
                        'open_interest': int(row.get('openInterest', 0)),
                        'implied_volatility': float(row.get('impliedVolatility', 0))
                    })
                    
            return options_list
            
        except Exception as e:
            logger.error(f"Yahoo options error for {symbol}: {e}")
            return []
            
    async def get_options_chain(self, symbol: str) -> List[OptionsData]:
        """Get options chain asynchronously"""
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(self.executor, self._fetch_options_sync, symbol)
        
        return [OptionsData(**opt) for opt in data]

class PolygonIOSource(DataSourceBase):
    """Polygon.io data source (professional)"""
    
    def __init__(self, api_key: str):
        super().__init__("PolygonIO")
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.ws_url = "wss://socket.polygon.io"
        self.ws = None
        
    async def connect(self):
        """Connect to Polygon websocket"""
        try:
            # For real implementation, use polygon-api-client
            self.is_connected = True
            logger.info(f"✅ {self.name} connected")
        except Exception as e:
            logger.error(f"Polygon connection error: {e}")
            self.is_connected = False
            
    async def disconnect(self):
        """Disconnect websocket"""
        if self.ws:
            self.ws.close()
        self.is_connected = False
        
    async def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote from Polygon"""
        try:
            url = f"{self.base_url}/v2/last/nbbo/{symbol}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # In production, use aiohttp or requests
            # For now, simplified implementation
            response = requests.get(url, headers=headers)
            data = response.json()
                    
            if data['status'] == 'OK':
                result = data['results']
                return MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(result['t'] / 1000),
                    price=(result['P'] + result['p']) / 2,  # Mid price
                    bid=result['p'],
                    ask=result['P'],
                    volume=0,  # Would need trades endpoint
                    high=0,
                    low=0,
                    open=0,
                    close=0
                )
        except Exception as e:
            logger.error(f"Polygon quote error: {e}")
            
        return None

class AlpacaDataSource(DataSourceBase):
    """Alpaca Markets data source"""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        super().__init__("Alpaca")
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        
    async def connect(self):
        """Initialize Alpaca connection"""
        # Would use alpaca-py in production
        self.is_connected = True
        logger.info(f"✅ {self.name} connected")
        
    async def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get quote from Alpaca"""
        # Implementation would use alpaca-py
        pass

class UnifiedDataFeed:
    """Unified data feed combining multiple sources"""
    
    def __init__(self, primary_source: str = "yahoo"):
        self.sources: Dict[str, DataSourceBase] = {}
        self.primary_source = primary_source
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.data_cache: Dict[str, MarketData] = {}
        self.options_cache: Dict[str, List[OptionsData]] = {}
        self.running = False
        
    def add_source(self, name: str, source: DataSourceBase):
        """Add a data source"""
        self.sources[name] = source
        
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for data events"""
        self.callbacks[event_type].append(callback)
        
    async def connect_all(self):
        """Connect all data sources"""
        tasks = []
        for name, source in self.sources.items():
            tasks.append(source.connect())
            
        await asyncio.gather(*tasks)
        logger.info("✅ All data sources connected")
        
    async def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get quote with failover"""
        # Try primary source first
        if self.primary_source in self.sources:
            data = await self.sources[self.primary_source].get_quote(symbol)
            if data:
                self.data_cache[symbol] = data
                await self._notify_callbacks('quote', data)
                return data
                
        # Failover to other sources
        for name, source in self.sources.items():
            if name != self.primary_source and source.is_connected:
                data = await source.get_quote(symbol)
                if data:
                    self.data_cache[symbol] = data
                    await self._notify_callbacks('quote', data)
                    return data
                    
        # Return cached data if available
        return self.data_cache.get(symbol)
        
    async def get_options_chain(self, symbol: str) -> List[OptionsData]:
        """Get options chain with failover"""
        for name, source in self.sources.items():
            if source.is_connected:
                try:
                    data = await source.get_options_chain(symbol)
                    if data:
                        self.options_cache[symbol] = data
                        await self._notify_callbacks('options', data)
                        return data
                except Exception as e:
                    logger.error(f"Options chain error from {name}: {e}")
                    
        return self.options_cache.get(symbol, [])
        
    async def _notify_callbacks(self, event_type: str, data: Any):
        """Notify registered callbacks"""
        for callback in self.callbacks[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                
    async def start_streaming(self, symbols: List[str], interval: int = 1):
        """Start streaming data for symbols"""
        self.running = True
        
        while self.running:
            tasks = []
            
            # Fetch quotes for all symbols
            for symbol in symbols:
                tasks.append(self.get_quote(symbol))
                
            # Fetch options (less frequently)
            if int(datetime.now().timestamp()) % 60 == 0:  # Every minute
                for symbol in symbols:
                    tasks.append(self.get_options_chain(symbol))
                    
            await asyncio.gather(*tasks)
            await asyncio.sleep(interval)
            
    def stop_streaming(self):
        """Stop streaming data"""
        self.running = False

class AgentDataAdapter:
    """Adapts market data for agent consumption"""
    
    def __init__(self, data_bus):
        self.data_bus = data_bus
        self.price_history: Dict[str, List[MarketData]] = defaultdict(list)
        self.options_history: Dict[str, List[OptionsData]] = defaultdict(list)
        
    async def on_quote_update(self, data: MarketData):
        """Handle quote updates"""
        symbol = data.symbol
        
        # Update history
        self.price_history[symbol].append(data)
        self.price_history[symbol] = self.price_history[symbol][-100:]  # Keep last 100
        
        # Create DataFrame for technical analysis
        df = pd.DataFrame([asdict(d) for d in self.price_history[symbol]])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Publish to data bus
        self.data_bus.publish('price_action', {
            'symbol': symbol,
            'current_price': data.price,
            'bid': data.bid,
            'ask': data.ask,
            'spread': data.ask - data.bid,
            'data_frame': df
        })
        
        # Volume analysis
        self.data_bus.publish('volume', {
            'symbol': symbol,
            'current_volume': data.volume,
            'vwap': data.vwap,
            'trades_count': data.trades_count
        })
        
    async def on_options_update(self, options: List[OptionsData]):
        """Handle options updates"""
        if not options:
            return
            
        symbol = options[0].symbol
        
        # Calculate aggregate metrics
        calls = [opt for opt in options if opt.option_type == 'call']
        puts = [opt for opt in options if opt.option_type == 'put']
        
        # Put/Call ratio
        call_volume = sum(opt.volume for opt in calls)
        put_volume = sum(opt.volume for opt in puts)
        pc_ratio = put_volume / call_volume if call_volume > 0 else 1.0
        
        # Unusual activity detection
        unusual_options = []
        for opt in options:
            if opt.volume > opt.open_interest * 0.5 and opt.volume > 100:
                unusual_options.append({
                    'strike': opt.strike,
                    'type': opt.option_type,
                    'volume': opt.volume,
                    'oi': opt.open_interest,
                    'iv': opt.implied_volatility
                })
                
        # Publish to data bus
        self.data_bus.publish('options_flow', {
            'symbol': symbol,
            'put_call_ratio': pc_ratio,
            'total_call_volume': call_volume,
            'total_put_volume': put_volume,
            'unusual_activity': unusual_options,
            'options_chain': options
        })

# Example usage
async def main():
    """Example of setting up live data feed"""
    from agents.common.data_bus import AgentDataBus

    # Initialize components
    data_bus = AgentDataBus()
    adapter = AgentDataAdapter(data_bus)
    feed = UnifiedDataFeed(primary_source="yahoo")
    
    # Add data sources
    yahoo_source = YahooFinanceSource()
    feed.add_source("yahoo", yahoo_source)
    
    # Add Polygon if API key available
    if os.getenv('POLYGON_API_KEY'):
        polygon_source = PolygonIOSource(os.getenv('POLYGON_API_KEY'))
        feed.add_source("polygon", polygon_source)
    
    # Register callbacks
    feed.register_callback('quote', adapter.on_quote_update)
    feed.register_callback('options', adapter.on_options_update)
    
    # Connect sources
    await feed.connect_all()
    
    # Start streaming
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'SPY']
    await feed.start_streaming(symbols)

if __name__ == "__main__":
    asyncio.run(main()) 