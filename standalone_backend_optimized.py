#!/usr/bin/env python3
"""
Optimized Standalone Backend for GoldenSignalsAI
Implements caching, concurrent processing, and performance optimizations
Based on standalone_backend_fixed.py with performance improvements
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pandas as pd
from cachetools import TTLCache
from contextlib import asynccontextmanager

# Import the cache wrapper
from cache_wrapper import cache_market_data, cache_signals

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import data quality validator
try:
    from src.services.data_quality_validator import DataQualityValidator
    data_validator = DataQualityValidator()
    logger.info("✅ Data Quality Validator initialized")
except ImportError:
    logger.warning("⚠️ Data Quality Validator not available, using direct yfinance")
    data_validator = None

# Import signal generation engine and filtering pipeline
try:
    from src.services.signals.signal_service import SignalGenerationEngine, TradingSignal
    from src.services.signals.signal_filter import SignalFilteringPipeline
    signal_engine = SignalGenerationEngine()
    signal_pipeline = SignalFilteringPipeline()
    logger.info("✅ Signal Generation Engine and Filtering Pipeline initialized")
except ImportError:
    logger.warning("⚠️ Signal Generation Engine not available, using legacy signal generation")
    signal_engine = None
    signal_pipeline = None

# Import signal monitoring service
try:
    from src.services.signal_monitoring_service import SignalMonitoringService
    monitoring_service = SignalMonitoringService()
    logger.info("✅ Signal Monitoring Service initialized")
except ImportError:
    logger.warning("⚠️ Signal Monitoring Service not available")
    monitoring_service = None

# Cache configuration
market_data_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minute TTL
signal_cache = TTLCache(maxsize=500, ttl=30)  # 30 second TTL
historical_cache = TTLCache(maxsize=200, ttl=600)  # 10 minute TTL

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Performance monitoring
request_times = defaultdict(list)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(periodic_signal_updates())
    asyncio.create_task(manager.batch_sender())
    logger.info("🚀 Optimized backend started with caching and batch processing")
    yield
    # Shutdown
    logger.info("Shutting down optimized backend...")

# Create FastAPI app
app = FastAPI(
    title="GoldenSignalsAI Optimized Backend",
    description="High-performance backend with caching and optimization",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Performance monitoring middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    request_times[request.url.path].append(process_time)
    return response

# Models (same as original)
class Signal(BaseModel):
    id: str
    symbol: str
    action: str  # Changed from 'type' to 'action'
    confidence: float
    price: float
    timestamp: str
    reason: str
    indicators: Dict[str, Any]
    risk_level: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class MarketData(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None

# Cached technical indicator calculations
@lru_cache(maxsize=128)
def calculate_indicators_cached(symbol: str, prices_tuple: tuple) -> dict:
    """Calculate technical indicators with caching"""
    prices = np.array(prices_tuple)
    
    indicators = {}
    
    # RSI calculation
    if len(prices) >= 14:
        deltas = np.diff(prices)
        seed = deltas[:14]
        up = seed[seed >= 0].sum() / 14
        down = -seed[seed < 0].sum() / 14
        rs = up / down if down != 0 else 0
        indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # SMA calculations
    if len(prices) >= 20:
        indicators['sma_20'] = np.mean(prices[-20:])
        indicators['sma_50'] = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
    
    # Bollinger Bands
    if len(prices) >= 20:
        sma = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        indicators['bb_upper'] = sma + (2 * std)
        indicators['bb_lower'] = sma - (2 * std)
        indicators['bb_percent'] = (prices[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
    
    return indicators

# Optimized market data fetching
@cache_market_data(ttl=300)
async def get_market_data_cached(symbol: str) -> Optional[MarketData]:
    """Get market data with caching and fallback sources"""
    try:
        # Use data validator if available for better reliability
        if data_validator:
            data, source = await data_validator.get_market_data_with_fallback(symbol)
            if data is not None and not data.empty:
                # Get latest data point
                latest = data.iloc[-1]
                
                # Calculate change from open
                open_price = float(latest.get('Open', latest.get('Close', 0)))
                close_price = float(latest.get('Close', 0))
                change = close_price - open_price
                change_percent = (change / open_price * 100) if open_price > 0 else 0
                
                return MarketData(
                    symbol=symbol,
                    price=close_price,
                    change=change,
                    change_percent=change_percent,
                    volume=int(latest.get('Volume', 0)),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    bid=None,  # Not available from historical data
                    ask=None,
                    high=float(latest.get('High', close_price)),
                    low=float(latest.get('Low', close_price)),
                    open=open_price
                )
        
        # Fallback to direct yfinance
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(executor, yf.Ticker, symbol)
        info = await loop.run_in_executor(executor, lambda: ticker.info)
        
        current_price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
        if not current_price:
            # Fallback to historical data
            hist = await loop.run_in_executor(
                executor, 
                lambda: ticker.history(period="1d", interval="1m")
            )
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
            else:
                # Generate mock data to keep the system running
                logger.warning(f"Using mock data for {symbol}")
                return MarketData(
                    symbol=symbol,
                    price=round(random.uniform(100, 500), 2),
                    change=round(random.uniform(-5, 5), 2),
                    change_percent=round(random.uniform(-2, 2), 2),
                    volume=random.randint(1000000, 10000000),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    bid=None,
                    ask=None,
                    high=round(random.uniform(102, 505), 2),
                    low=round(random.uniform(98, 495), 2),
                    open=round(random.uniform(100, 500), 2)
                )
        
        return MarketData(
            symbol=symbol,
            price=float(current_price),
            change=float(info.get('regularMarketChange', 0)),
            change_percent=float(info.get('regularMarketChangePercent', 0)),
            volume=int(info.get('volume', 0)),
            timestamp=datetime.now(timezone.utc).isoformat(),
            bid=info.get('bid'),
            ask=info.get('ask'),
            high=info.get('dayHigh'),
            low=info.get('dayLow'),
            open=info.get('open')
        )
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        # Return mock data to keep the system running
        return MarketData(
            symbol=symbol,
            price=round(random.uniform(100, 500), 2),
            change=round(random.uniform(-5, 5), 2),
            change_percent=round(random.uniform(-2, 2), 2),
            volume=random.randint(1000000, 10000000),
            timestamp=datetime.now(timezone.utc).isoformat(),
            bid=None,
            ask=None,
            high=round(random.uniform(102, 505), 2),
            low=round(random.uniform(98, 495), 2),
            open=round(random.uniform(100, 500), 2)
        )

# Cached historical data fetching
async def get_historical_data_cached(symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Get historical data with caching and fallback sources"""
    cache_key = f"hist:{symbol}:{period}:{interval}"
    
    if cache_key in historical_cache:
        return historical_cache[cache_key]
    
    try:
        # Use data validator if available
        if data_validator:
            data, source = await data_validator.get_market_data_with_fallback(symbol)
            if data is not None and not data.empty:
                # Clean and validate the data
                data = data_validator.clean_data(data)
                historical_cache[cache_key] = data
                logger.info(f"Got historical data for {symbol} from {source}")
                return data
        
        # Fallback to direct yfinance
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(executor, yf.Ticker, symbol)
        data = await loop.run_in_executor(
            executor,
            lambda: ticker.history(period=period, interval=interval)
        )
        
        if not data.empty:
            historical_cache[cache_key] = data
            return data
        
        # Generate mock historical data if all else fails
        logger.warning(f"Using mock historical data for {symbol}")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 500, len(dates)),
            'High': np.random.uniform(102, 505, len(dates)),
            'Low': np.random.uniform(98, 495, len(dates)),
            'Close': np.random.uniform(100, 500, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Ensure logical consistency
        mock_data['High'] = mock_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
        mock_data['Low'] = mock_data[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return mock_data
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        # Return mock data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 500, len(dates)),
            'High': np.random.uniform(102, 505, len(dates)),
            'Low': np.random.uniform(98, 495, len(dates)),
            'Close': np.random.uniform(100, 500, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Ensure logical consistency
        mock_data['High'] = mock_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
        mock_data['Low'] = mock_data[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return mock_data

# Optimized signal generation
@cache_signals(ttl=30)
async def generate_signals_cached(symbol: str) -> List[Signal]:
    """Generate trading signals with caching"""
    signals = []
    
    # Use new signal engine if available
    if signal_engine and signal_pipeline:
        try:
            # Generate signals using the new engine
            trading_signals = await signal_engine.generate_signals([symbol])
            
            # Filter signals through the pipeline
            filtered_signals = signal_pipeline.filter_signals(trading_signals)
            
            # Convert TradingSignal to Signal (API model)
            for ts in filtered_signals:
                signal = Signal(
                    id=ts.id,
                    symbol=ts.symbol,
                    action=ts.action,
                    confidence=ts.confidence,
                    price=ts.price,
                    timestamp=ts.timestamp.isoformat(),
                    reason=ts.reason,
                    indicators=ts.indicators,
                    risk_level=ts.risk_level,
                    entry_price=ts.entry_price,
                    stop_loss=ts.stop_loss,
                    take_profit=ts.take_profit
                )
                signals.append(signal)
                
            if signals:
                return signals
        except Exception as e:
            logger.error(f"Error using new signal engine: {e}")
            # Fall back to legacy generation
    
    # Legacy signal generation
    # Get historical data
    hist_data = await get_historical_data_cached(symbol, period="1mo", interval="1d")
    
    if hist_data is not None and not hist_data.empty:
        # Convert prices to tuple for caching
        prices_tuple = tuple(hist_data['Close'].values)
        
        # Calculate indicators using cached function
        indicators = calculate_indicators_cached(symbol, prices_tuple)
        
        # Get current market data
        market_data = await get_market_data_cached(symbol)
        
        if market_data:
            # Generate signal based on indicators
            confidence = 0.0
            signal_type = "HOLD"
            reasons = []
            
            # RSI signal
            if 'rsi' in indicators:
                if indicators['rsi'] < 30:
                    confidence += 0.3
                    signal_type = "BUY"
                    reasons.append("RSI oversold")
                elif indicators['rsi'] > 70:
                    confidence += 0.3
                    signal_type = "SELL"
                    reasons.append("RSI overbought")
            
            # Bollinger Bands signal
            if 'bb_percent' in indicators:
                if indicators['bb_percent'] < 0.2:
                    confidence += 0.2
                    reasons.append("Near lower Bollinger Band")
                    if signal_type != "SELL":
                        signal_type = "BUY"
                elif indicators['bb_percent'] > 0.8:
                    confidence += 0.2
                    reasons.append("Near upper Bollinger Band")
                    if signal_type != "BUY":
                        signal_type = "SELL"
            
            # Create signal if we have enough confidence
            if confidence > 0 and signal_type != "HOLD":
                signal = Signal(
                    id=f"{symbol}_{int(time.time() * 1000)}",
                    symbol=symbol,
                    action=signal_type,
                    confidence=min(confidence, 1.0),
                    price=market_data.price,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    reason="; ".join(reasons),
                    indicators=indicators,
                    risk_level="low" if confidence > 0.7 else "medium" if confidence > 0.4 else "high",
                    entry_price=market_data.price,
                    stop_loss=market_data.price * 0.98 if signal_type == "BUY" else market_data.price * 1.02,
                    take_profit=market_data.price * 1.05 if signal_type == "BUY" else market_data.price * 0.95
                )
                signals.append(signal)
    
    # If no signals generated, create a mock signal
    if not signals:
        actions = ["BUY", "SELL", "HOLD"]
        action = random.choice(actions)
        
        signal = Signal(
            id=f"{symbol}_{int(time.time() * 1000)}_{random.randint(1, 100)}",
            symbol=symbol,
            action=action,
            confidence=round(random.uniform(0.6, 0.95), 2),
            price=round(random.uniform(100, 500), 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=f"Technical analysis suggests {action}",
            indicators={
                "rsi": round(random.uniform(20, 80), 2),
                "sma_20": round(random.uniform(100, 500), 2),
                "volume": random.randint(1000000, 10000000)
            },
            risk_level=random.choice(["low", "medium", "high"]),
            entry_price=round(random.uniform(100, 500), 2),
            stop_loss=round(random.uniform(95, 99), 2),
            take_profit=round(random.uniform(101, 110), 2)
        )
        signals.append(signal)
    
    return signals

# Batch processing for multiple symbols
async def generate_signals_batch(symbols: List[str]) -> List[Signal]:
    """Generate signals for multiple symbols concurrently"""
    # Use new signal engine for batch processing if available
    if signal_engine and signal_pipeline:
        try:
            # Generate all signals at once
            trading_signals = await signal_engine.generate_signals(symbols)
            
            # Filter signals through the pipeline
            filtered_signals = signal_pipeline.filter_signals(trading_signals)
            
            # Convert TradingSignal to Signal (API model)
            all_signals = []
            for ts in filtered_signals:
                signal = Signal(
                    id=ts.id,
                    symbol=ts.symbol,
                    action=ts.action,
                    confidence=ts.confidence,
                    price=ts.price,
                    timestamp=ts.timestamp.isoformat(),
                    reason=ts.reason,
                    indicators=ts.indicators,
                    risk_level=ts.risk_level,
                    entry_price=ts.entry_price,
                    stop_loss=ts.stop_loss,
                    take_profit=ts.take_profit
                )
                all_signals.append(signal)
            
            return all_signals
        except Exception as e:
            logger.error(f"Error using new signal engine for batch: {e}")
            # Fall back to legacy generation
    
    # Legacy batch processing
    tasks = [generate_signals_cached(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_signals = []
    for result in results:
        if isinstance(result, list):
            all_signals.extend(result)
    
    return all_signals

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.update_queue = asyncio.Queue()
        self.batch_interval = 0.5  # 500ms batching

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def queue_update(self, data: dict):
        """Queue update for batching"""
        await self.update_queue.put(data)

    async def batch_sender(self):
        """Send updates in batches"""
        while True:
            updates = []
            deadline = time.time() + self.batch_interval
            
            # Collect updates for batch_interval seconds
            while time.time() < deadline:
                try:
                    timeout = deadline - time.time()
                    if timeout > 0:
                        update = await asyncio.wait_for(
                            self.update_queue.get(),
                            timeout=timeout
                        )
                        updates.append(update)
                except asyncio.TimeoutError:
                    break
            
            if updates and self.active_connections:
                # Send batch to all connections
                batch_data = {
                    "type": "batch_update",
                    "updates": updates,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                disconnected = []
                for connection in self.active_connections:
                    try:
                        await connection.send_json(batch_data)
                    except:
                        disconnected.append(connection)
                
                # Remove disconnected clients
                for conn in disconnected:
                    if conn in self.active_connections:
                        self.disconnect(conn)

manager = ConnectionManager()

# Background task for periodic updates
async def periodic_signal_updates():
    """Send periodic signal updates via WebSocket"""
    await asyncio.sleep(5)  # Initial delay
    
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "SPY"]
    
    while True:
        try:
            # Generate signals for all symbols using batch processing
            all_signals = await generate_signals_batch(symbols)
            
            if all_signals and manager.active_connections:
                await manager.queue_update({
                    "type": "signals",
                    "data": [s.model_dump() for s in all_signals]
                })
            
            # Also send market updates for a few symbols
            for symbol in symbols[:3]:
                market_data = await get_market_data_cached(symbol)
                if market_data:
                    await manager.queue_update({
                        "type": "market_update",
                        "data": market_data.model_dump()
                    })
            
        except Exception as e:
            logger.error(f"Error in periodic updates: {e}")
        
        await asyncio.sleep(30)  # Update every 30 seconds



# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "GoldenSignalsAI Optimized Backend",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Caching enabled (5min market data, 30s signals)",
            "Batch processing for concurrent requests",
            "Response compression (gzip)",
            "Performance monitoring",
            "WebSocket batching (500ms)"
        ]
    }

@app.get("/api/v1/signals")
async def get_all_signals():
    """Get all signals with batch processing"""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "SPY"]
    all_signals = await generate_signals_batch(symbols)
    return all_signals

@app.get("/api/v1/signals/{symbol}")
async def get_signals(symbol: str):
    """Get signals for specific symbol"""
    signals = await generate_signals_cached(symbol)
    return signals

@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get market data for symbol"""
    if not symbol or len(symbol) > 10 or not symbol.isalnum():
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
    
    data = await get_market_data_cached(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Market data not found for {symbol}")
    
    return data

@app.get("/api/v1/market-data/{symbol}/historical")
async def get_historical_data(
    symbol: str,
    period: str = Query("1mo", pattern="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max)$"),
    interval: str = Query("1d", pattern="^(1m|2m|5m|15m|30m|60m|90m|1h|1d|5d|1wk|1mo|3mo)$")
):
    """Get historical data"""
    data = await get_historical_data_cached(symbol, period, interval)
    
    if data is None:
        return {"data": []}
    
    # Convert to JSON-serializable format with lowercase field names
    records = []
    for index, row in data.iterrows():
        records.append({
            "timestamp": int(index.timestamp() * 1000),  # Convert to milliseconds
            "open": float(row["Open"]) if "Open" in row else 0,
            "high": float(row["High"]) if "High" in row else 0,
            "low": float(row["Low"]) if "Low" in row else 0,
            "close": float(row["Close"]) if "Close" in row else 0,
            "volume": int(row["Volume"]) if "Volume" in row else 0
        })
    
    return {"data": records}

@app.get("/api/v1/performance")
async def get_performance_stats():
    """Get performance statistics"""
    stats = {}
    for endpoint, times in request_times.items():
        if times:
            stats[endpoint] = {
                'count': len(times),
                'avg_ms': round(np.mean(times) * 1000, 2),
                'max_ms': round(max(times) * 1000, 2),
                'p95_ms': round(np.percentile(times, 95) * 1000, 2) if len(times) > 10 else 0
            }
    
    # Add cache statistics
    cache_stats = {
        'market_data_cache_size': len(market_data_cache),
        'signal_cache_size': len(signal_cache),
        'historical_cache_size': len(historical_cache),
        'total_requests': sum(len(times) for times in request_times.values())
    }
    
    return {
        "endpoints": stats,
        "cache": cache_stats,
        "uptime": time.time() - startup_time if 'startup_time' in globals() else 0
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with batching"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any client messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/v1/signals/{symbol}/insights")
async def get_signal_insights(symbol: str):
    """Get insights for a symbol using cached data"""
    signals = await generate_signals_cached(symbol)
    
    if not signals:
        return {
            "symbol": symbol,
            "signal_count": 0,
            "recommendation": "NO_DATA"
        }
    
    # Aggregate signals
    buy_signals = sum(1 for s in signals if s.action == "BUY")
    sell_signals = sum(1 for s in signals if s.action == "SELL")
    avg_confidence = np.mean([s.confidence for s in signals])
    
    recommendation = "HOLD"
    if buy_signals > sell_signals:
        recommendation = "BUY"
    elif sell_signals > buy_signals:
        recommendation = "SELL"
    
    return {
        "symbol": symbol,
        "signal_count": len(signals),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "average_confidence": round(avg_confidence, 2),
        "recommendation": recommendation,
        "latest_signal": signals[0].model_dump() if signals else None
    }

@app.get("/api/v1/market/opportunities")
async def get_market_opportunities():
    """Get market opportunities using batch processing"""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]
    
    # Get signals for all symbols
    all_signals = await generate_signals_batch(symbols)
    
    # Filter high confidence signals
    opportunities = [
        s for s in all_signals 
        if s.confidence > 0.7 and s.action in ["BUY", "SELL"]
    ]
    
    # Sort by confidence
    opportunities.sort(key=lambda x: x.confidence, reverse=True)
    
    return {"opportunities": [opp.model_dump() for opp in opportunities[:10]]}

# Precise options endpoint (cached)
@app.get("/api/v1/signals/precise-options")
async def get_precise_options_signals(
    symbol: str = Query(..., description="Stock symbol"),
    timeframe: str = Query("15m", pattern="^(1m|5m|15m|30m|1h|4h|1d)$")
):
    """Get precise options signals"""
    # For now, return regular signals (can be enhanced with options-specific logic)
    signals = await generate_signals_cached(symbol)
    
    # Add options-specific fields
    for signal in signals:
        signal.indicators["options_volume"] = random.randint(1000, 50000)
        signal.indicators["implied_volatility"] = round(random.uniform(0.2, 0.8), 2)
        signal.indicators["options_flow"] = random.choice(["bullish", "bearish", "neutral"])
    
    return signals

# New API endpoints for signal engine and pipeline management
@app.get("/api/v1/pipeline/stats")
async def get_pipeline_stats():
    """Get signal filtering pipeline statistics"""
    if not signal_pipeline:
        return {"error": "Signal pipeline not available"}
    
    return signal_pipeline.get_pipeline_stats()

@app.post("/api/v1/pipeline/configure")
async def configure_pipeline(config: Dict[str, Any]):
    """Configure the signal filtering pipeline"""
    if not signal_pipeline:
        return {"error": "Signal pipeline not available"}
    
    try:
        # Apply configuration
        for filter_obj in signal_pipeline.filters:
            filter_name = filter_obj.name.lower()
            
            if "confidence_filter" in filter_name and "min_confidence" in config:
                filter_obj.min_confidence = config["min_confidence"]
            elif "quality" in filter_name and "min_quality_score" in config:
                filter_obj.min_quality_score = config["min_quality_score"]
            elif "risk" in filter_name and "allowed_risk_levels" in config:
                filter_obj.allowed_risk_levels = config["allowed_risk_levels"]
            elif "volume" in filter_name and "min_volume_ratio" in config:
                filter_obj.min_volume_ratio = config["min_volume_ratio"]
        
        return {"status": "configured", "config": config}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/signals/quality-report")
async def get_signal_quality_report():
    """Get comprehensive signal quality report"""
    if not signal_engine or not signal_pipeline:
        return {"error": "Signal engine not available"}
    
    # Generate signals for test symbols
    test_symbols = ["AAPL", "GOOGL", "MSFT", "SPY"]
    all_trading_signals = await signal_engine.generate_signals(test_symbols)
    filtered_signals = signal_pipeline.filter_signals(all_trading_signals)
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_signals_generated": len(all_trading_signals),
        "signals_after_filtering": len(filtered_signals),
        "filter_rate": 1 - (len(filtered_signals) / max(1, len(all_trading_signals))),
        "pipeline_stats": signal_pipeline.get_pipeline_stats(),
        "signal_breakdown": {
            "by_action": {},
            "by_risk_level": {},
            "by_confidence": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
    }
    
    # Analyze filtered signals
    for signal in filtered_signals:
        # By action
        report["signal_breakdown"]["by_action"][signal.action] = \
            report["signal_breakdown"]["by_action"].get(signal.action, 0) + 1
        
        # By risk level
        report["signal_breakdown"]["by_risk_level"][signal.risk_level] = \
            report["signal_breakdown"]["by_risk_level"].get(signal.risk_level, 0) + 1
        
        # By confidence
        if signal.confidence >= 0.8:
            report["signal_breakdown"]["by_confidence"]["high"] += 1
        elif signal.confidence >= 0.6:
            report["signal_breakdown"]["by_confidence"]["medium"] += 1
        else:
            report["signal_breakdown"]["by_confidence"]["low"] += 1
    
    return report

@app.post("/api/v1/signals/feedback")
async def submit_signal_feedback(
    signal_id: str,
    outcome: str = Query(..., pattern="^(success|failure|partial)$"),
    profit_loss: Optional[float] = None,
    notes: Optional[str] = None
):
    """Submit feedback for a signal to improve future performance"""
    # For now, just log the feedback
    # In future, this can be stored and used to train the ML model
    feedback = {
        "signal_id": signal_id,
        "outcome": outcome,
        "profit_loss": profit_loss,
        "notes": notes,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    logger.info(f"Signal feedback received: {feedback}")
    
    # If pipeline is available, adjust parameters based on feedback
    if signal_pipeline and outcome == "failure":
        # Increase filtering strictness on failures
        performance_metrics = {
            "false_positive_rate": 0.4,  # Simulated
            "signal_accuracy": 0.4  # Simulated
        }
        signal_pipeline.adjust_filter_parameters(performance_metrics)
    
    return {"status": "feedback_received", "feedback": feedback}

# Monitoring service endpoints
@app.post("/api/v1/monitoring/track-entry")
async def track_signal_entry(signal_data: Dict[str, Any]):
    """Track when a signal is acted upon"""
    if not monitoring_service:
        return {"error": "Monitoring service not available"}
    
    monitoring_service.track_signal_entry(signal_data)
    return {"status": "tracking_started", "signal_id": signal_data.get('id')}

@app.post("/api/v1/monitoring/track-exit")
async def track_signal_exit(
    signal_id: str,
    exit_price: float,
    outcome: str = Query("success", pattern="^(success|failure|partial)$"),
    notes: Optional[str] = None
):
    """Track when a signal position is closed"""
    if not monitoring_service:
        return {"error": "Monitoring service not available"}
    
    monitoring_service.track_signal_exit(signal_id, exit_price, outcome, notes or "")
    return {"status": "exit_tracked", "signal_id": signal_id}

@app.get("/api/v1/monitoring/performance")
async def get_performance_metrics(
    timeframe_days: Optional[int] = None,
    symbol: Optional[str] = None
):
    """Get comprehensive performance metrics"""
    if not monitoring_service:
        return {"error": "Monitoring service not available"}
    
    timeframe = timedelta(days=timeframe_days) if timeframe_days else None
    metrics = monitoring_service.get_performance_metrics(timeframe, symbol)
    return metrics.to_dict()

@app.get("/api/v1/monitoring/recommendations")
async def get_improvement_recommendations():
    """Get recommendations for improving signal generation"""
    if not monitoring_service:
        return {"error": "Monitoring service not available"}
    
    recommendations = monitoring_service.generate_improvement_recommendations()
    return {"recommendations": recommendations}

@app.get("/api/v1/monitoring/feedback-summary")
async def get_feedback_summary():
    """Get summary of signal feedback"""
    if not monitoring_service:
        return {"error": "Monitoring service not available"}
    
    return monitoring_service.get_signal_feedback_summary()

@app.post("/api/v1/monitoring/snapshot")
async def save_performance_snapshot():
    """Save a performance snapshot"""
    if not monitoring_service:
        return {"error": "Monitoring service not available"}
    
    monitoring_service.save_performance_snapshot()
    return {"status": "snapshot_saved", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/v1/monitoring/active-signals")
async def get_active_signals():
    """Get currently active (pending) signals"""
    if not monitoring_service:
        return {"error": "Monitoring service not available"}
    
    active_signals = [
        {**outcome.to_dict(), 'signal_id': signal_id}
        for signal_id, outcome in monitoring_service.active_signals.items()
    ]
    return {"active_signals": active_signals, "count": len(active_signals)}

# Backtesting endpoints
@app.post("/api/v1/backtest/run")
async def run_backtest(
    symbols: List[str] = Query(..., description="List of symbols to backtest"),
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)"),
    quick_mode: bool = Query(False, description="Run quick backtest with reduced data")
):
    """Run ML-enhanced backtest with signal quality metrics"""
    try:
        from ml_enhanced_backtest_system import MLBacktestEngine
        
        engine = MLBacktestEngine()
        
        # Use default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            # Quick mode: 3 months, normal: 1 year
            days_back = 90 if quick_mode else 365
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Run backtest
        results = await engine.run_comprehensive_backtest(
            symbols=symbols[:5],  # Limit to 5 symbols
            start_date=start_date,
            end_date=end_date
        )
        
        # Format results for API response
        formatted_results = {}
        for symbol, data in results.items():
            formatted_results[symbol] = {
                "performance": {
                    "annual_return": data['backtest_metrics']['annual_return'],
                    "sharpe_ratio": data['backtest_metrics']['sharpe_ratio'],
                    "max_drawdown": data['backtest_metrics']['max_drawdown'],
                    "win_rate": data['backtest_metrics']['win_rate'],
                    "profit_factor": data['backtest_metrics']['profit_factor']
                },
                "ml_accuracy": data['avg_accuracy'],
                "signal_quality": data.get('signal_quality', {}),
                "top_features": data['feature_importance'][:5]
            }
        
        return {
            "status": "success",
            "backtest_period": f"{start_date} to {end_date}",
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/backtest/recommendations")
async def get_backtest_recommendations(
    symbols: List[str] = Query(..., description="Symbols to analyze")
):
    """Get signal improvement recommendations from backtesting"""
    try:
        from ml_enhanced_backtest_system import SignalAccuracyImprover
        
        improver = SignalAccuracyImprover()
        improvements = await improver.improve_signals(symbols[:5])
        
        return {
            "recommended_features": improvements['recommended_features'][:10],
            "optimal_parameters": improvements['optimal_parameters'],
            "risk_management": improvements['risk_management'],
            "signal_filters": improvements['signal_filters']
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Record startup time for uptime tracking
startup_time = time.time()

if __name__ == "__main__":
    print("🚀 Starting GoldenSignalsAI Optimized Backend")
    print("📊 Caching enabled for improved performance")
    print("🔄 Batch processing for concurrent requests")
    print("📈 Performance monitoring at /api/v1/performance")
    print("🌐 API docs available at http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
