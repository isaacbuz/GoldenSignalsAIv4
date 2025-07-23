"""
Live Data Service for GoldenSignalsAI V3
Integrates multiple real-time data sources with fallback mechanisms
"""

import asyncio
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import jso
import numpy as np
import pandas as pd
import timezonen
import websockets
import yfinance as yf
from agents.common.data_bus import AgentDataBus
from aiohttp import ClientSession

from src.core.redis_manager import RedisManager
from src.data.fetchers.live_data_fetcher import (
    AgentDataAdapter,
    MarketData,
    OptionsData,
    PolygonIOSource,
    UnifiedDataFeed,
    YahooFinanceSource,
)
from src.websocket.manager import WebSocketManager

logger = logging.getLogger(__name__)

@dataclass
class LiveDataConfig:
    """Configuration for live data service"""
    primary_source: str = "yahoo"
    enable_polygon: bool = True
    enable_alpaca: bool = False
    enable_ib: bool = False
    update_interval: int = 5  # seconds
    options_update_interval: int = 60  # seconds
    cache_ttl: int = 300  # seconds
    max_retry_attempts: int = 3
    symbols: List[str] = None

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['AAPL', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'NVDA', 'META', 'AMZN']

class LiveDataService:
    """
    Main service for live market data integration
    Handles multiple data sources, caching, and distribution
    """
    
    def __init__(self, config: LiveDataConfig = None):
        self.config = config or LiveDataConfig()
        self.data_feed = UnifiedDataFeed(primary_source=self.config.primary_source)
        self.data_bus = AgentDataBus()
        self.adapter = AgentDataAdapter(self.data_bus)
        self.redis_manager = RedisManager()
        self.ws_manager = WebSocketManager()
        
        self.running = False
        self.tasks = []
        
        # Performance tracking
        self.stats = {
            "quotes_fetched": 0,
            "options_fetched": 0,
            "errors": 0,
            "last_update": None,
            "uptime_start": None
        }
        
    async def initialize(self):
        """Initialize all data sources"""
        logger.info("🚀 Initializing Live Data Service...")
        
        # Always add Yahoo Finance (free)
        yahoo_source = YahooFinanceSource()
        self.data_feed.add_source("yahoo", yahoo_source)
        logger.info("✅ Added Yahoo Finance data source")
        
        # Add Polygon if enabled and API key available
        if self.config.enable_polygon:
            polygon_key = os.getenv('POLYGON_API_KEY')
            if polygon_key:
                polygon_source = PolygonIOSource(polygon_key)
                self.data_feed.add_source("polygon", polygon_source)
                logger.info("✅ Added Polygon.io data source")
            else:
                logger.warning("⚠️ POLYGON_API_KEY not found - Polygon disabled")
        
        # Register callbacks
        self.data_feed.register_callback('quote', self._on_quote_update)
        self.data_feed.register_callback('options', self._on_options_update)
        
        # Connect all sources
        await self.data_feed.connect_all()
        
        # Initialize Redis connection
        await self.redis_manager.initialize()
        
        self.stats["uptime_start"] = datetime.now(timezone.utc)
        logger.info("✅ Live Data Service initialized")
        
    async def start(self):
        """Start the live data service"""
        if self.running:
            logger.warning("Live Data Service already running")
            return
            
        self.running = True
        logger.info(f"📡 Starting live data streaming for {len(self.config.symbols)} symbols")
        
        # Start main tasks
        self.tasks = [
            asyncio.create_task(self._quote_update_loop()),
            asyncio.create_task(self._options_update_loop()),
            asyncio.create_task(self._websocket_broadcast_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._cache_cleanup_loop()),
        ]
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Live Data Service tasks cancelled")
        except Exception as e:
            logger.error(f"Live Data Service error: {e}")
            
    async def stop(self):
        """Stop the live data service"""
        logger.info("🛑 Stopping Live Data Service...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        # Stop data feed
        self.data_feed.stop_streaming()
        
        # Disconnect sources
        for source in self.data_feed.sources.values():
            await source.disconnect()
            
        logger.info("✅ Live Data Service stopped")
        
    async def _quote_update_loop(self):
        """Main loop for fetching quote updates"""
        while self.running:
            try:
                # Fetch quotes for all symbols
                tasks = []
                for symbol in self.config.symbols:
                    tasks.append(self.data_feed.get_quote(symbol))
                    
                quotes = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process successful quotes
                for quote in quotes:
                    if isinstance(quote, MarketData):
                        self.stats["quotes_fetched"] += 1
                        await self._process_quote(quote)
                    elif isinstance(quote, Exception):
                        self.stats["errors"] += 1
                        logger.error(f"Quote fetch error: {quote}")
                        
                self.stats["last_update"] = datetime.now(timezone.utc)
                
            except Exception as e:
                logger.error(f"Quote update loop error: {e}")
                self.stats["errors"] += 1
                
            await asyncio.sleep(self.config.update_interval)
            
    async def _options_update_loop(self):
        """Loop for fetching options data"""
        while self.running:
            try:
                # Fetch options for all symbols
                tasks = []
                for symbol in self.config.symbols:
                    tasks.append(self.data_feed.get_options_chain(symbol))
                    
                options_chains = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process successful options data
                for i, chain in enumerate(options_chains):
                    if isinstance(chain, list) and chain:
                        symbol = self.config.symbols[i]
                        self.stats["options_fetched"] += 1
                        await self._process_options(symbol, chain)
                    elif isinstance(chain, Exception):
                        self.stats["errors"] += 1
                        logger.error(f"Options fetch error: {chain}")
                        
            except Exception as e:
                logger.error(f"Options update loop error: {e}")
                self.stats["errors"] += 1
                
            await asyncio.sleep(self.config.options_update_interval)
            
    async def _websocket_broadcast_loop(self):
        """Broadcast updates to WebSocket clients"""
        while self.running:
            try:
                # Get active WebSocket connections
                active_connections = self.ws_manager.get_active_connections()
                
                if active_connections:
                    # Prepare market summary
                    summary = await self._prepare_market_summary()
                    
                    # Broadcast to all connections
                    await self.ws_manager.broadcast({
                        "type": "market_summary",
                        "data": summary,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")
                
            await asyncio.sleep(10)  # Broadcast every 10 seconds
            
    async def _health_check_loop(self):
        """Monitor service health"""
        while self.running:
            try:
                # Check data source health
                for name, source in self.data_feed.sources.items():
                    if not source.is_connected:
                        logger.warning(f"Data source {name} disconnected, attempting reconnect...")
                        await source.connect()
                        
                # Log statistics
                uptime = datetime.now(timezone.utc) - self.stats["uptime_start"]
                logger.info(
                    f"📊 Live Data Stats - Uptime: {uptime}, "
                    f"Quotes: {self.stats['quotes_fetched']}, "
                    f"Options: {self.stats['options_fetched']}, "
                    f"Errors: {self.stats['errors']}"
                )
                
                # Publish health status
                await self.redis_manager.publish_market_data("system", {
                    "service": "live_data",
                    "status": "healthy",
                    "stats": self.stats,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                
            await asyncio.sleep(60)  # Check every minute
            
    async def _cache_cleanup_loop(self):
        """Clean up old cached data"""
        while self.running:
            try:
                # Clean up old cache entries
                cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.config.cache_ttl)
                
                # This would clean up Redis cache
                # Implementation depends on specific caching strategy
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                
            await asyncio.sleep(300)  # Clean every 5 minutes
            
    async def _on_quote_update(self, quote: MarketData):
        """Handle quote updates from data feed"""
        try:
            # Update adapter (publishes to data bus)
            await self.adapter.on_quote_update(quote)
            
            # Cache in Redis
            await self.redis_manager.cache_market_data(
                quote.symbol,
                asdict(quote),
                ttl=self.config.cache_ttl
            )
            
            # Publish to Redis pub/sub
            await self.redis_manager.publish_market_data(quote.symbol, asdict(quote))
            
            # Log significant price movements
            if abs(quote.price - quote.open) / quote.open > 0.02:  # 2% move
                logger.info(
                    f"🚨 Significant move - {quote.symbol}: "
                    f"${quote.price:.2f} ({((quote.price - quote.open) / quote.open * 100):.2f}%)"
                )
                
        except Exception as e:
            logger.error(f"Quote update error for {quote.symbol}: {e}")
            
    async def _on_options_update(self, options: List[OptionsData]):
        """Handle options updates from data feed"""
        try:
            if not options:
                return
                
            # Update adapter
            await self.adapter.on_options_update(options)
            
            # Cache in Redis
            symbol = options[0].symbol
            options_data = [asdict(opt) for opt in options]
            await self.redis_manager.cache_options_data(
                symbol,
                options_data,
                ttl=self.config.cache_ttl
            )
            
            # Detect unusual options activity
            unusual = self._detect_unusual_options_activity(options)
            if unusual:
                logger.info(f"🎯 Unusual options activity detected for {symbol}")
                await self.redis_manager.publish_signal(symbol, {
                    "type": "unusual_options",
                    "data": unusual,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
        except Exception as e:
            logger.error(f"Options update error: {e}")
            
    async def _process_quote(self, quote: MarketData):
        """Process individual quote"""
        # Additional processing if needed
        pass
        
    async def _process_options(self, symbol: str, options: List[OptionsData]):
        """Process options chain"""
        # Additional processing if needed
        pass
        
    async def _prepare_market_summary(self) -> Dict[str, Any]:
        """Prepare market summary for broadcast"""
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": {},
            "indices": {},
            "stats": self.stats
        }
        
        # Get latest data for each symbol
        for symbol in self.config.symbols:
            cached_data = await self.redis_manager.get_cached_market_data(symbol)
            if cached_data:
                summary["symbols"][symbol] = {
                    "price": cached_data.get("price"),
                    "change": cached_data.get("price", 0) - cached_data.get("open", 0),
                    "change_percent": ((cached_data.get("price", 0) - cached_data.get("open", 0)) / 
                                     cached_data.get("open", 1) * 100) if cached_data.get("open") else 0,
                    "volume": cached_data.get("volume"),
                    "bid": cached_data.get("bid"),
                    "ask": cached_data.get("ask")
                }
                
        return summary
        
    def _detect_unusual_options_activity(self, options: List[OptionsData]) -> Optional[List[Dict]]:
        """Detect unusual options activity"""
        unusual = []
        
        for opt in options:
            # High volume relative to open interest
            if opt.volume > opt.open_interest * 0.5 and opt.volume > 100:
                unusual.append({
                    "strike": opt.strike,
                    "type": opt.option_type,
                    "expiration": opt.expiration,
                    "volume": opt.volume,
                    "open_interest": opt.open_interest,
                    "volume_oi_ratio": opt.volume / opt.open_interest if opt.open_interest > 0 else 0,
                    "implied_volatility": opt.implied_volatility
                })
                
        return unusual if unusual else None
        
    async def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get latest quote for symbol"""
        # Try cache first
        cached = await self.redis_manager.get_cached_market_data(symbol)
        if cached and (datetime.now(timezone.utc) - datetime.fromisoformat(cached.get("timestamp", "")) < timedelta(seconds=10)):
            return MarketData(**cached)
            
        # Fetch fresh data
        return await self.data_feed.get_quote(symbol)
        
    async def get_options_chain(self, symbol: str) -> List[OptionsData]:
        """Get options chain for symbol"""
        # Try cache first
        cached = await self.redis_manager.get_cached_options_data(symbol)
        if cached:
            return [OptionsData(**opt) for opt in cached]
            
        # Fetch fresh data
        return await self.data_feed.get_options_chain(symbol)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self.stats,
            "uptime": str(datetime.now(timezone.utc) - self.stats["uptime_start"]) if self.stats["uptime_start"] else "0:00:00",
            "data_sources": list(self.data_feed.sources.keys()),
            "active_symbols": self.config.symbols
        }

# Singleton instance
_live_data_service: Optional[LiveDataService] = None

def get_live_data_service(config: LiveDataConfig = None) -> LiveDataService:
    """Get or create live data service instance"""
    global _live_data_service
    if _live_data_service is None:
        _live_data_service = LiveDataService(config)
    return _live_data_service

async def start_live_data_service(config: LiveDataConfig = None):
    """Start the live data service"""
    service = get_live_data_service(config)
    await service.initialize()
    await service.start() 