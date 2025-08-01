"""
Professional WebSocket Service for GoldenSignals
Implements multi-source data aggregation with fallback
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources"""

    ALPACA = "alpaca"
    POLYGON = "polygon"
    FINNHUB = "finnhub"
    YAHOO = "yahoo"
    CACHE = "cache"


@dataclass
class MarketData:
    """Standardized market data structure"""

    symbol: str
    price: float
    volume: int
    high: float
    low: float
    open: float
    close: float
    timestamp: datetime
    source: DataSource
    bid: Optional[float] = None
    ask: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "close": self.close,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "bid": self.bid,
            "ask": self.ask,
        }


@dataclass
class SourceConfig:
    """Configuration for a data source"""

    name: DataSource
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    rate_limits: Dict[str, int] = field(default_factory=dict)
    priority: int = 0  # Lower is higher priority
    enabled: bool = True


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate: float, per: float):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()

    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        current = time.time()
        time_passed = current - self.last_check
        self.last_check = current
        self.allowance += time_passed * (self.rate / self.per)

        if self.allowance > self.rate:
            self.allowance = self.rate

        if self.allowance < 1.0:
            sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
            await asyncio.sleep(sleep_time)
            self.allowance = 0.0
        else:
            self.allowance -= 1.0


class DataCache:
    """Smart caching for market data"""

    def __init__(self, ttl_seconds: int = 60):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache: Dict[str, tuple[MarketData, datetime]] = {}

    def get(self, symbol: str) -> Optional[MarketData]:
        """Get cached data if still valid"""
        if symbol in self.cache:
            data, timestamp = self.cache[symbol]
            if datetime.now() - timestamp < self.ttl:
                return data
        return None

    def set(self, symbol: str, data: MarketData):
        """Cache market data"""
        self.cache[symbol] = (data, datetime.now())

    def clear_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired = [
            symbol for symbol, (_, timestamp) in self.cache.items() if now - timestamp >= self.ttl
        ]
        for symbol in expired:
            del self.cache[symbol]


class BaseConnector:
    """Base class for data source connectors"""

    def __init__(self, config: SourceConfig):
        self.config = config
        self.rate_limiter = None
        if config.rate_limits:
            # Use the most restrictive rate limit
            if "per_second" in config.rate_limits:
                self.rate_limiter = RateLimiter(config.rate_limits["per_second"], 1.0)
            elif "per_minute" in config.rate_limits:
                self.rate_limiter = RateLimiter(config.rate_limits["per_minute"], 60.0)

    async def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get current quote for symbol"""
        raise NotImplementedError

    async def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to real-time updates"""
        raise NotImplementedError

    async def unsubscribe(self, symbol: str):
        """Unsubscribe from updates"""
        raise NotImplementedError

    async def connect(self):
        """Establish connection"""
        pass

    async def disconnect(self):
        """Close connection"""
        pass

    async def _rate_limit(self):
        """Apply rate limiting if configured"""
        if self.rate_limiter:
            await self.rate_limiter.acquire()


class AlpacaConnector(BaseConnector):
    """Alpaca Markets connector"""

    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self.base_url = "https://data.alpaca.markets/v2"
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        self.session = None
        self.ws = None
        self.subscriptions = {}

    async def connect(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            headers={
                "APCA-API-KEY-ID": self.config.api_key,
                "APCA-API-SECRET-KEY": self.config.secret_key,
            }
        )

    async def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get latest quote via REST API"""
        await self._rate_limit()

        try:
            url = f"{self.base_url}/stocks/{symbol}/quotes/latest"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    quote = data.get("quote", {})

                    return MarketData(
                        symbol=symbol,
                        price=(quote.get("bp", 0) + quote.get("ap", 0)) / 2,
                        volume=0,
                        high=0,
                        low=0,
                        open=0,
                        close=0,
                        timestamp=datetime.fromisoformat(
                            quote.get("t", datetime.now().isoformat())
                        ),
                        source=DataSource.ALPACA,
                        bid=quote.get("bp"),
                        ask=quote.get("ap"),
                    )
        except Exception as e:
            logger.error(f"Alpaca quote error for {symbol}: {e}")

        return None

    async def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to WebSocket updates"""
        self.subscriptions[symbol] = callback

        if not self.ws:
            await self._connect_websocket()

        # Send subscription message
        await self.ws.send_json({"action": "subscribe", "bars": [symbol]})

    async def _connect_websocket(self):
        """Connect to Alpaca WebSocket"""
        self.ws = await aiohttp.ClientSession().ws_connect(self.ws_url)

        # Authenticate
        await self.ws.send_json(
            {"action": "auth", "key": self.config.api_key, "secret": self.config.secret_key}
        )

        # Start message handler
        asyncio.create_task(self._handle_messages())

    async def _handle_messages(self):
        """Process WebSocket messages"""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)

                for item in data:
                    if item.get("T") == "b":  # Bar data
                        symbol = item.get("S")
                        if symbol in self.subscriptions:
                            market_data = MarketData(
                                symbol=symbol,
                                price=item.get("c"),
                                volume=item.get("v"),
                                high=item.get("h"),
                                low=item.get("l"),
                                open=item.get("o"),
                                close=item.get("c"),
                                timestamp=datetime.fromisoformat(item.get("t")),
                                source=DataSource.ALPACA,
                            )
                            await self.subscriptions[symbol](market_data)


class FinnhubConnector(BaseConnector):
    """Finnhub connector"""

    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self.base_url = "https://finnhub.io/api/v1"
        self.ws_url = f"wss://ws.finnhub.io?token={config.api_key}"
        self.session = None
        self.ws = None
        self.subscriptions = {}

    async def connect(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()

    async def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get latest quote via REST API"""
        await self._rate_limit()

        try:
            url = f"{self.base_url}/quote"
            params = {"symbol": symbol, "token": self.config.api_key}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    return MarketData(
                        symbol=symbol,
                        price=data.get("c", 0),
                        volume=0,
                        high=data.get("h", 0),
                        low=data.get("l", 0),
                        open=data.get("o", 0),
                        close=data.get("c", 0),
                        timestamp=datetime.now(),
                        source=DataSource.FINNHUB,
                    )
        except Exception as e:
            logger.error(f"Finnhub quote error for {symbol}: {e}")

        return None

    async def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to WebSocket updates"""
        self.subscriptions[symbol] = callback

        if not self.ws:
            await self._connect_websocket()

        # Send subscription message
        await self.ws.send_json({"type": "subscribe", "symbol": symbol})

    async def _connect_websocket(self):
        """Connect to Finnhub WebSocket"""
        self.ws = await aiohttp.ClientSession().ws_connect(self.ws_url)
        asyncio.create_task(self._handle_messages())

    async def _handle_messages(self):
        """Process WebSocket messages"""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)

                if data.get("type") == "trade":
                    for trade in data.get("data", []):
                        symbol = trade.get("s")
                        if symbol in self.subscriptions:
                            market_data = MarketData(
                                symbol=symbol,
                                price=trade.get("p"),
                                volume=trade.get("v"),
                                high=trade.get("p"),
                                low=trade.get("p"),
                                open=trade.get("p"),
                                close=trade.get("p"),
                                timestamp=datetime.fromtimestamp(trade.get("t", 0) / 1000),
                                source=DataSource.FINNHUB,
                            )
                            await self.subscriptions[symbol](market_data)


class ProfessionalWebSocketService:
    """
    Multi-source WebSocket service with automatic fallback
    """

    def __init__(self):
        self.sources: List[BaseConnector] = []
        self.cache = DataCache(ttl_seconds=30)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.running = False
        self._tasks = []

    def add_source(self, config: SourceConfig, connector_class: type):
        """Add a data source"""
        if config.enabled:
            connector = connector_class(config)
            self.sources.append(connector)
            self.sources.sort(key=lambda x: x.config.priority)

    async def start(self):
        """Start the service"""
        self.running = True

        # Connect all sources
        for source in self.sources:
            try:
                await source.connect()
                logger.info(f"Connected to {source.config.name.value}")
            except Exception as e:
                logger.error(f"Failed to connect to {source.config.name.value}: {e}")

        # Start cache cleanup task
        self._tasks.append(asyncio.create_task(self._cache_cleanup_task()))

        # Start health check task
        self._tasks.append(asyncio.create_task(self._health_check_task()))

    async def stop(self):
        """Stop the service"""
        self.running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        # Disconnect sources
        for source in self.sources:
            try:
                await source.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {source.config.name.value}: {e}")

    async def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get quote with automatic fallback"""
        # Check cache first
        cached = self.cache.get(symbol)
        if cached:
            logger.debug(f"Returning cached data for {symbol}")
            return cached

        # Try each source in priority order
        for source in self.sources:
            try:
                data = await source.get_quote(symbol)
                if data:
                    self.cache.set(symbol, data)
                    return data
            except Exception as e:
                logger.warning(f"Source {source.config.name.value} failed for {symbol}: {e}")
                continue

        logger.error(f"All sources failed for {symbol}")
        return None

    async def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to real-time updates"""
        self.subscribers[symbol].append(callback)

        # Subscribe to primary source
        if self.sources:
            try:
                await self.sources[0].subscribe(
                    symbol, lambda data: self._handle_update(symbol, data)
                )
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")

    async def unsubscribe(self, symbol: str):
        """Unsubscribe from updates"""
        if symbol in self.subscribers:
            del self.subscribers[symbol]

        # Unsubscribe from all sources
        for source in self.sources:
            try:
                await source.unsubscribe(symbol)
            except Exception:
                pass

    async def _handle_update(self, symbol: str, data: MarketData):
        """Handle incoming market data update"""
        # Update cache
        self.cache.set(symbol, data)

        # Notify subscribers
        for callback in self.subscribers.get(symbol, []):
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")

    async def _cache_cleanup_task(self):
        """Periodically clean up expired cache entries"""
        while self.running:
            self.cache.clear_expired()
            await asyncio.sleep(60)  # Run every minute

    async def _health_check_task(self):
        """Monitor source health"""
        while self.running:
            for source in self.sources:
                try:
                    # Simple health check - try to get a common symbol
                    await asyncio.wait_for(source.get_quote("AAPL"), timeout=5.0)
                except Exception as e:
                    logger.warning(f"Health check failed for {source.config.name.value}: {e}")
            await asyncio.sleep(30)  # Check every 30 seconds

    def is_market_open(self) -> bool:
        """Check if US market is currently open"""
        ny_tz = pytz.timezone("America/New_York")
        now = datetime.now(ny_tz)

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        # Check if weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        return market_open <= now <= market_close


# Example usage
async def main():
    """Example of using the professional WebSocket service"""

    # Configure sources
    service = ProfessionalWebSocketService()

    # Add Alpaca (primary)
    service.add_source(
        SourceConfig(
            name=DataSource.ALPACA,
            api_key="YOUR_ALPACA_KEY",
            secret_key="YOUR_ALPACA_SECRET",
            rate_limits={"per_minute": 200},
            priority=0,
        ),
        AlpacaConnector,
    )

    # Add Finnhub (secondary)
    service.add_source(
        SourceConfig(
            name=DataSource.FINNHUB,
            api_key="YOUR_FINNHUB_KEY",
            rate_limits={"per_minute": 60},
            priority=1,
        ),
        FinnhubConnector,
    )

    # Start service
    await service.start()

    # Define callback
    async def on_update(data: MarketData):
        print(f"Update: {data.symbol} @ ${data.price} from {data.source.value}")

    # Subscribe to symbols
    await service.subscribe("AAPL", on_update)
    await service.subscribe("GOOGL", on_update)

    # Get quotes
    quote = await service.get_quote("MSFT")
    if quote:
        print(f"Quote: {quote.symbol} @ ${quote.price}")

    # Keep running
    try:
        await asyncio.sleep(300)  # Run for 5 minutes
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
