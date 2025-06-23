# GoldenSignalsAI V2 - Live Data Integration Guide

## 🌐 Overview

To make GoldenSignalsAI work with real market data, we need to integrate multiple data sources for different types of information. This guide covers how to connect to live data feeds and feed them to our agents.

## 📊 Data Requirements

### Essential Data Types

1. **Real-time Stock Prices** (for all agents)
   - Current price, bid/ask spreads
   - Volume data
   - High/low/open/close

2. **Options Data** (for options agents)
   - Options chains with bid/ask
   - Implied volatility
   - Greeks calculations
   - Open interest and volume

3. **Market Depth** (for volume agents)
   - Level 2 order book data
   - Time & sales
   - Block trades

4. **News & Sentiment** (for sentiment agents)
   - Real-time news feeds
   - Social media sentiment
   - Analyst ratings

## 🔌 Data Source Options

### 1. Free Data Sources (Limited but Good for Testing)

#### Yahoo Finance (yfinance)
```python
# Already partially implemented in market_data_service.py
import yfinance as yf

class YahooDataFetcher:
    def __init__(self, symbols):
        self.symbols = symbols
    
    def get_real_time_quote(self, symbol):
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'symbol': symbol,
            'price': info.get('regularMarketPrice', 0),
            'bid': info.get('bid', 0),
            'ask': info.get('ask', 0),
            'volume': info.get('volume', 0),
            'previousClose': info.get('previousClose', 0)
        }
    
    def get_options_chain(self, symbol):
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        options_data = []
        for exp in expirations[:3]:  # First 3 expirations
            opt_chain = ticker.option_chain(exp)
            
            # Process calls
            for _, row in opt_chain.calls.iterrows():
                options_data.append({
                    'type': 'call',
                    'strike': row['strike'],
                    'expiration': exp,
                    'bid': row['bid'],
                    'ask': row['ask'],
                    'volume': row['volume'],
                    'openInterest': row['openInterest'],
                    'impliedVolatility': row['impliedVolatility']
                })
        
        return options_data
```

#### Alpha Vantage (Free tier: 5 API calls/minute)
```python
# pip install alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

class AlphaVantageDataFetcher:
    def __init__(self, api_key):
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.ti = TechIndicators(key=api_key, output_format='pandas')
    
    def get_intraday_data(self, symbol):
        data, meta = self.ts.get_intraday(
            symbol=symbol, 
            interval='1min', 
            outputsize='compact'
        )
        return data
    
    def get_technical_indicators(self, symbol):
        # Get RSI
        rsi, _ = self.ti.get_rsi(symbol=symbol, interval='1min')
        # Get MACD
        macd, _ = self.ti.get_macd(symbol=symbol, interval='1min')
        return {'rsi': rsi, 'macd': macd}
```

### 2. Professional Data Sources (Paid but Comprehensive)

#### Polygon.io (Recommended for Production)
```python
# pip install polygon-api-client
from polygon import RESTClient
from polygon.websocket import WebSocketClient, Market

class PolygonDataFetcher:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)
        self.ws_client = None
        self.callbacks = {}
    
    def setup_websocket(self, on_message_callback):
        """Setup real-time websocket feed"""
        self.ws_client = WebSocketClient(
            Market.Stocks,
            api_key=self.api_key,
            on_message=on_message_callback
        )
    
    def subscribe_to_symbols(self, symbols):
        """Subscribe to real-time data for symbols"""
        if self.ws_client:
            for symbol in symbols:
                # Subscribe to trades
                self.ws_client.subscribe(f"T.{symbol}")
                # Subscribe to quotes
                self.ws_client.subscribe(f"Q.{symbol}")
                # Subscribe to aggregate bars
                self.ws_client.subscribe(f"A.{symbol}")
    
    def get_options_chain(self, symbol, expiration_date):
        """Get options chain data"""
        options = self.client.list_options_contracts(
            underlying_ticker=symbol,
            expiration_date=expiration_date,
            limit=1000
        )
        
        chain_data = []
        for contract in options:
            # Get quotes for each contract
            quotes = self.client.get_last_quote(
                contract.ticker
            )
            
            chain_data.append({
                'ticker': contract.ticker,
                'strike': contract.strike_price,
                'type': contract.contract_type,
                'expiration': contract.expiration_date,
                'bid': quotes.bid,
                'ask': quotes.ask,
                'volume': quotes.day_volume,
                'open_interest': quotes.open_interest
            })
        
        return chain_data
    
    def get_market_snapshot(self, symbols):
        """Get snapshot of multiple symbols"""
        snapshots = self.client.get_snapshot_all("stocks")
        
        data = {}
        for symbol in symbols:
            if symbol in snapshots:
                snap = snapshots[symbol]
                data[symbol] = {
                    'price': snap.day.close,
                    'volume': snap.day.volume,
                    'change': snap.day.close - snap.prev_day.close,
                    'change_percent': snap.percent_change
                }
        
        return data
```

#### Interactive Brokers API
```python
# pip install ib_insync
from ib_insync import IB, Stock, Option, util

class IBDataFetcher:
    def __init__(self):
        self.ib = IB()
        
    def connect(self, host='127.0.0.1', port=7497, clientId=1):
        """Connect to TWS or IB Gateway"""
        self.ib.connect(host, port, clientId)
    
    def get_real_time_bars(self, symbol):
        """Subscribe to real-time 5-second bars"""
        contract = Stock(symbol, 'SMART', 'USD')
        bars = self.ib.reqRealTimeBars(
            contract, 5, 'TRADES', False
        )
        return bars
    
    def get_market_depth(self, symbol):
        """Get Level 2 market depth"""
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.reqMktDepth(contract)
        
        # Return order book
        return {
            'bids': contract.domBids,
            'asks': contract.domAsks
        }
    
    def get_options_chain(self, symbol, expiration):
        """Get options chain with Greeks"""
        # Get all strikes for expiration
        chains = self.ib.reqSecDefOptParams(
            symbol, '', 'STK', 100
        )
        
        options_data = []
        for chain in chains:
            if expiration in chain.expirations:
                for strike in chain.strikes:
                    # Create option contracts
                    call = Option(symbol, expiration, strike, 'C', 'SMART')
                    put = Option(symbol, expiration, strike, 'P', 'SMART')
                    
                    # Get market data and Greeks
                    self.ib.reqMktData(call, '', False, False)
                    self.ib.reqMktData(put, '', False, False)
                    
                    # Wait for data
                    self.ib.sleep(0.5)
                    
                    options_data.append({
                        'strike': strike,
                        'call': {
                            'bid': call.bid,
                            'ask': call.ask,
                            'delta': call.delta,
                            'gamma': call.gamma,
                            'theta': call.theta,
                            'vega': call.vega,
                            'iv': call.implVol
                        },
                        'put': {
                            'bid': put.bid,
                            'ask': put.ask,
                            'delta': put.delta,
                            'gamma': put.gamma,
                            'theta': put.theta,
                            'vega': put.vega,
                            'iv': put.implVol
                        }
                    })
        
        return options_data
```

#### TDAmeritrade API
```python
# pip install tda-api
import tda
from tda import auth, client

class TDAmeritradeDataFetcher:
    def __init__(self, api_key, redirect_uri, token_path):
        self.api_key = api_key
        self.redirect_uri = redirect_uri
        self.token_path = token_path
        self.client = self._authenticate()
    
    def _authenticate(self):
        """Authenticate with TD Ameritrade"""
        try:
            c = auth.client_from_token_file(
                self.token_path, 
                self.api_key
            )
        except FileNotFoundError:
            # First time authentication
            c = auth.client_from_manual_flow(
                self.api_key, 
                self.redirect_uri, 
                self.token_path
            )
        return c
    
    def get_quotes(self, symbols):
        """Get real-time quotes"""
        response = self.client.get_quotes(symbols)
        quotes = response.json()
        
        data = {}
        for symbol, quote in quotes.items():
            data[symbol] = {
                'price': quote['lastPrice'],
                'bid': quote['bidPrice'],
                'ask': quote['askPrice'],
                'volume': quote['totalVolume'],
                'high': quote['highPrice'],
                'low': quote['lowPrice']
            }
        
        return data
    
    def get_options_chain(self, symbol):
        """Get options chain with Greeks"""
        response = self.client.get_option_chain(
            symbol,
            contract_type=client.Options.ContractType.ALL,
            include_quotes=True
        )
        
        chain = response.json()
        return self._process_options_chain(chain)
```

### 3. News & Sentiment Data

#### NewsAPI
```python
# pip install newsapi-python
from newsapi import NewsApiClient

class NewsDataFetcher:
    def __init__(self, api_key):
        self.newsapi = NewsApiClient(api_key=api_key)
    
    def get_company_news(self, symbol, company_name):
        """Get news for specific company"""
        articles = self.newsapi.get_everything(
            q=f'"{company_name}" OR {symbol}',
            language='en',
            sort_by='publishedAt',
            page_size=20
        )
        
        return [{
            'title': article['title'],
            'description': article['description'],
            'url': article['url'],
            'publishedAt': article['publishedAt'],
            'source': article['source']['name'],
            'sentiment': self._analyze_sentiment(article['title'])
        } for article in articles['articles']]
```

#### Reddit API (via PRAW)
```python
# pip install praw
import praw

class RedditSentimentFetcher:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def get_wsb_sentiment(self, symbol):
        """Get sentiment from WallStreetBets"""
        subreddit = self.reddit.subreddit('wallstreetbets')
        
        mentions = 0
        sentiment_scores = []
        
        # Check hot posts
        for post in subreddit.hot(limit=100):
            if symbol in post.title.upper() or symbol in post.selftext.upper():
                mentions += 1
                # Simple sentiment based on upvote ratio
                sentiment_scores.append(post.upvote_ratio)
        
        return {
            'mentions': mentions,
            'avg_sentiment': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5,
            'trending': mentions > 10
        }
```

## 🔄 Real-time Data Pipeline

### Unified Data Fetcher
```python
import asyncio
from typing import Dict, List, Callable
import pandas as pd

class UnifiedDataFetcher:
    """Combines multiple data sources into unified feed"""
    
    def __init__(self):
        self.sources = {}
        self.callbacks = {}
        self.running = False
        
    def add_source(self, name: str, fetcher):
        """Add a data source"""
        self.sources[name] = fetcher
        
    def register_callback(self, data_type: str, callback: Callable):
        """Register callback for specific data type"""
        if data_type not in self.callbacks:
            self.callbacks[data_type] = []
        self.callbacks[data_type].append(callback)
    
    async def fetch_and_distribute(self):
        """Main loop to fetch and distribute data"""
        while self.running:
            tasks = []
            
            # Fetch from all sources
            for name, source in self.sources.items():
                if hasattr(source, 'get_real_time_data'):
                    tasks.append(self._fetch_source(name, source))
            
            # Wait for all fetches
            results = await asyncio.gather(*tasks)
            
            # Distribute to callbacks
            for result in results:
                if result:
                    await self._distribute_data(result)
            
            # Wait before next fetch
            await asyncio.sleep(1)  # 1 second interval
    
    async def _fetch_source(self, name: str, source):
        """Fetch from individual source"""
        try:
            data = await source.get_real_time_data()
            return {'source': name, 'data': data}
        except Exception as e:
            print(f"Error fetching from {name}: {e}")
            return None
    
    async def _distribute_data(self, result):
        """Distribute data to registered callbacks"""
        source = result['source']
        data = result['data']
        
        # Determine data type
        if 'options' in source:
            data_type = 'options'
        elif 'news' in source:
            data_type = 'sentiment'
        else:
            data_type = 'market'
        
        # Call registered callbacks
        if data_type in self.callbacks:
            for callback in self.callbacks[data_type]:
                try:
                    await callback(data)
                except Exception as e:
                    print(f"Error in callback: {e}")
```

## 🤖 Feeding Data to Agents

### Agent Data Adapter
```python
class AgentDataAdapter:
    """Adapts raw market data for agent consumption"""
    
    def __init__(self, data_bus):
        self.data_bus = data_bus
        self.data_buffer = {}
        
    def process_market_data(self, raw_data: Dict):
        """Process market data for agents"""
        symbol = raw_data['symbol']
        
        # Update buffer
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        self.data_buffer[symbol].append({
            'timestamp': raw_data['timestamp'],
            'price': raw_data['price'],
            'volume': raw_data['volume'],
            'bid': raw_data['bid'],
            'ask': raw_data['ask']
        })
        
        # Keep only last 100 data points
        self.data_buffer[symbol] = self.data_buffer[symbol][-100:]
        
        # Convert to DataFrame for agents
        df = pd.DataFrame(self.data_buffer[symbol])
        
        # Publish to data bus
        self.data_bus.publish('price_action', {
            'symbol': symbol,
            'current_price': raw_data['price'],
            'data_frame': df,
            'spread': raw_data['ask'] - raw_data['bid']
        })
        
        # Publish volume data
        self.data_bus.publish('volume', {
            'symbol': symbol,
            'current_volume': raw_data['volume'],
            'volume_profile': self._calculate_volume_profile(df)
        })
    
    def process_options_data(self, options_data: Dict):
        """Process options data for options agents"""
        # Calculate aggregated metrics
        put_call_ratio = self._calculate_put_call_ratio(options_data)
        gamma_exposure = self._calculate_gamma_exposure(options_data)
        
        # Publish to data bus
        self.data_bus.publish('options_flow', {
            'symbol': options_data['symbol'],
            'put_call_ratio': put_call_ratio,
            'gamma_exposure': gamma_exposure,
            'unusual_activity': self._detect_unusual_activity(options_data)
        })
```

## 🚀 Implementation Steps

### 1. Set Up Data Sources
```python
# config/data_sources.py
DATA_SOURCES = {
    'polygon': {
        'api_key': os.getenv('POLYGON_API_KEY'),
        'enabled': True,
        'rate_limit': 100  # requests per minute
    },
    'yahoo': {
        'enabled': True,
        'rate_limit': 2000  # requests per hour
    },
    'alpaca': {
        'api_key': os.getenv('ALPACA_API_KEY'),
        'secret_key': os.getenv('ALPACA_SECRET_KEY'),
        'enabled': False,
        'paper_trading': True
    },
    'newsapi': {
        'api_key': os.getenv('NEWS_API_KEY'),
        'enabled': True,
        'rate_limit': 500  # requests per day
    }
}
```

### 2. Create Data Service
```python
# infrastructure/data/live_data_service.py
class LiveDataService:
    def __init__(self, config):
        self.config = config
        self.fetchers = {}
        self.unified_fetcher = UnifiedDataFetcher()
        self.data_bus = AgentDataBus()
        self.adapter = AgentDataAdapter(self.data_bus)
        
        self._initialize_fetchers()
        
    def _initialize_fetchers(self):
        """Initialize enabled data fetchers"""
        if self.config['polygon']['enabled']:
            self.fetchers['polygon'] = PolygonDataFetcher(
                self.config['polygon']['api_key']
            )
            
        if self.config['yahoo']['enabled']:
            self.fetchers['yahoo'] = YahooDataFetcher(
                symbols=['AAPL', 'GOOGL', 'TSLA']  # etc
            )
    
    async def start(self):
        """Start live data service"""
        # Register data processors
        self.unified_fetcher.register_callback(
            'market', 
            self.adapter.process_market_data
        )
        self.unified_fetcher.register_callback(
            'options', 
            self.adapter.process_options_data
        )
        
        # Start fetching
        await self.unified_fetcher.fetch_and_distribute()
```

### 3. Update Agents to Use Live Data
```python
# agents/technical/enhanced_rsi_agent.py
class EnhancedRSIAgent:
    def __init__(self, data_bus):
        self.data_bus = data_bus
        self.data_bus.subscribe('price_action', self.on_price_update)
        
    def on_price_update(self, data):
        """Handle real-time price updates"""
        symbol = data['symbol']
        df = data['data_frame']
        
        # Calculate RSI on live data
        signal = self.generate_signal_from_df(symbol, df)
        
        # Publish signal
        if signal:
            self.data_bus.publish('signals', {
                'agent': 'EnhancedRSI',
                'symbol': symbol,
                'signal': signal
            })
```

## 🔧 Environment Setup

### 1. Install Dependencies
```bash
pip install yfinance polygon-api-client alpaca-py newsapi-python praw ib_insync tda-api websocket-client pandas numpy
```

### 2. Set Environment Variables
```bash
# .env file
POLYGON_API_KEY=your_polygon_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
NEWS_API_KEY=your_newsapi_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret
TD_API_KEY=your_td_ameritrade_key
IB_GATEWAY_PORT=7497
```

### 3. Start Services
```python
# start_live_data.py
import asyncio
from src.data.live_data_service import LiveDataService
from config.data_sources import DATA_SOURCES

async def main():
    # Initialize service
    service = LiveDataService(DATA_SOURCES)
    
    # Start data collection
    print("🚀 Starting live data service...")
    await service.start()

if __name__ == "__main__":
    asyncio.run(main())
```

## 📊 Data Quality & Reliability

### 1. Data Validation
```python
class DataValidator:
    @staticmethod
    def validate_price_data(data):
        """Validate price data quality"""
        if data['bid'] > data['ask']:
            return False, "Invalid bid/ask spread"
        
        if data['price'] < 0:
            return False, "Negative price"
        
        if data['volume'] < 0:
            return False, "Negative volume"
        
        return True, "Valid"
    
    @staticmethod
    def validate_options_data(data):
        """Validate options data"""
        if data['strike'] <= 0:
            return False, "Invalid strike price"
        
        if data['iv'] < 0 or data['iv'] > 5:
            return False, "Invalid implied volatility"
        
        return True, "Valid"
```

### 2. Failover Strategy
```python
class DataSourceFailover:
    def __init__(self, primary_source, backup_sources):
        self.primary = primary_source
        self.backups = backup_sources
        self.current_source = primary_source
    
    async def fetch_with_failover(self, symbol):
        """Fetch data with automatic failover"""
        try:
            data = await self.current_source.fetch(symbol)
            return data
        except Exception as e:
            print(f"Primary source failed: {e}")
            
            # Try backup sources
            for backup in self.backups:
                try:
                    data = await backup.fetch(symbol)
                    self.current_source = backup
                    return data
                except:
                    continue
            
            raise Exception("All data sources failed")
```

## 🎯 Best Practices

1. **Rate Limiting**: Respect API rate limits
   ```python
   from ratelimit import limits, sleep_and_retry
   
   @sleep_and_retry
   @limits(calls=5, period=60)  # 5 calls per minute
   def fetch_data(symbol):
       return api.get_quote(symbol)
   ```

2. **Data Caching**: Cache frequently accessed data
   ```python
   from functools import lru_cache
   from datetime import datetime, timedelta
   
   @lru_cache(maxsize=1000)
   def get_cached_quote(symbol, timestamp):
       return fetch_quote(symbol)
   ```

3. **Error Handling**: Graceful degradation
   ```python
   try:
       live_data = await fetch_live_data(symbol)
   except DataUnavailable:
       # Fall back to last known good data
       live_data = get_cached_data(symbol)
   ```

4. **Data Normalization**: Consistent format across sources
   ```python
   def normalize_price_data(source_name, raw_data):
       normalizers = {
           'polygon': normalize_polygon_data,
           'yahoo': normalize_yahoo_data,
           'alpaca': normalize_alpaca_data
       }
       return normalizers[source_name](raw_data)
   ```

## 🚨 Monitoring & Alerts

### Data Quality Monitoring
```python
class DataQualityMonitor:
    def __init__(self):
        self.metrics = {
            'latency': [],
            'missing_data': 0,
            'invalid_data': 0
        }
    
    def check_data_health(self):
        """Check overall data health"""
        avg_latency = sum(self.metrics['latency']) / len(self.metrics['latency'])
        
        if avg_latency > 1000:  # 1 second
            self.send_alert("High data latency detected")
        
        if self.metrics['missing_data'] > 10:
            self.send_alert("Excessive missing data")
```

## 🎯 Next Steps

1. **Choose Data Sources**: Based on budget and needs
2. **Implement Adapters**: Create adapters for chosen sources
3. **Test with Paper Trading**: Use paper trading accounts first
4. **Monitor Performance**: Track latency and data quality
5. **Scale Gradually**: Start with a few symbols, then expand

The key is to start simple with free sources like Yahoo Finance, then upgrade to professional sources as the system proves its value. 