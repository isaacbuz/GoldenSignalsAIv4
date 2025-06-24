"""
GoldenSignalsAI V3 - Main FastAPI Application

A next-generation AI trading platform featuring:
- Advanced multi-agent system with CrewAI
- Real-time WebSocket data streaming
- Sophisticated signal fusion and consensus
- Enterprise-grade monitoring and observability
- Modern async architecture
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import sys
import os

import sentry_sdk
from fastapi import FastAPI, WebSocket, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from redis import Redis
from functools import lru_cache

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings
from src.core.database import DatabaseManager
from src.core.redis_manager import RedisManager
from src.core.logging_config import setup_logging
from agents.orchestrator import AgentOrchestrator
from src.api.v1 import api_router
from src.api.v1.websocket import router as websocket_router
from src.websocket.manager import WebSocketManager
from src.services.signal_service import SignalService
from src.services.market_data_service import MarketDataService
from src.services.live_data_service import LiveDataService, LiveDataConfig
from src.middleware.security import SecurityMiddleware
from src.middleware.monitoring import MonitoringMiddleware


# Initialize logging
logger = setup_logging()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize Redis
redis = Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True
)

# Cache decorator
def cache_response(expire_time_seconds: int = 300):
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached_result = redis.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # If not in cache, execute function
            result = await func(request, *args, **kwargs)
            
            # Store in cache
            redis.setex(
                cache_key,
                expire_time_seconds,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management
    
    Handles startup and shutdown of all services, connections, and background tasks.
    """
    logger.info("🚀 Starting GoldenSignalsAI V3...")
    
    try:
        # Initialize Sentry for error tracking
        if settings.monitoring.sentry_dsn:
            sentry_sdk.init(
                dsn=settings.monitoring.sentry_dsn,
                traces_sample_rate=0.1,
                environment=settings.environment
            )
            logger.info("✅ Sentry error tracking initialized")
        
        # Initialize database connections
        app.state.db_manager = DatabaseManager()
        await app.state.db_manager.initialize()
        logger.info("✅ Database connections established")
        
        # Initialize Redis for caching and real-time data
        app.state.redis_manager = RedisManager()
        await app.state.redis_manager.initialize()
        logger.info("✅ Redis connections established")
        
        # Initialize WebSocket manager for real-time communication
        app.state.websocket_manager = WebSocketManager(app.state.redis_manager)
        await app.state.websocket_manager.initialize()
        logger.info("✅ WebSocket manager initialized")
        
        # Initialize core services
        app.state.signal_service = SignalService(app.state.db_manager, app.state.redis_manager)
        app.state.market_data_service = MarketDataService()
        await app.state.signal_service.initialize()
        logger.info("✅ Core services initialized")
        
        # Initialize live data service
        live_data_config = LiveDataConfig(
            primary_source="yahoo",
            enable_polygon=True,
            symbols=['AAPL', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'NVDA', 'META', 'AMZN', 'MSFT']
        )
        app.state.live_data_service = LiveDataService(live_data_config)
        await app.state.live_data_service.initialize()
        logger.info("✅ Live data service initialized")
        
        # Initialize agent orchestrator
        app.state.agent_orchestrator = AgentOrchestrator(
            signal_service=app.state.signal_service,
            market_data_service=app.state.market_data_service,
            websocket_manager=app.state.websocket_manager
        )
        await app.state.agent_orchestrator.initialize()
        logger.info("✅ Agent orchestrator initialized")
        
        # Start background tasks
        asyncio.create_task(app.state.live_data_service.start())
        asyncio.create_task(app.state.agent_orchestrator.start_signal_generation())
        logger.info("✅ Background tasks started")
        
        logger.info("🎯 GoldenSignalsAI V3 is fully operational!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {str(e)}")
        raise
    
    finally:
        # Cleanup on shutdown
        logger.info("🛑 Shutting down GoldenSignalsAI V3...")
        
        if getattr(app.state, 'live_data_service', None):
            await app.state.live_data_service.stop()
            
        if getattr(app.state, 'agent_orchestrator', None):
            await app.state.agent_orchestrator.shutdown()
            
        if getattr(app.state, 'signal_service', None):
            await app.state.signal_service.shutdown()
            
        if getattr(app.state, 'websocket_manager', None):
            await app.state.websocket_manager.shutdown()
            
        if getattr(app.state, 'redis_manager', None):
            await app.state.redis_manager.close()
            
        if getattr(app.state, 'db_manager', None):
            await app.state.db_manager.close()
            
        logger.info("✅ Graceful shutdown completed")


# Import API documentation configuration
from src.api.docs import API_TITLE, API_DESCRIPTION, API_VERSION, TAGS_METADATA, custom_openapi_schema

# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    # Merge with custom schema
    custom_schema = custom_openapi_schema()
    openapi_schema.update(custom_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Add security middleware
app.add_middleware(SecurityMiddleware)

# Add monitoring middleware
app.add_middleware(MonitoringMiddleware)

# Add session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.security.secret_key,
    max_age=settings.security.access_token_expire_minutes * 60
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Prometheus metrics
if settings.monitoring.prometheus_enabled:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)


# API Routes
app.include_router(api_router, prefix="/api/v1")
app.include_router(websocket_router)  # WebSocket routes


@app.get("/", tags=["health"])
async def root():
    """Root endpoint with system information"""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "status": "operational",
        "features": [
            "Advanced Multi-Agent Trading System",
            "Real-time Signal Generation",
            "WebSocket Data Streaming",
            "Enterprise Monitoring",
            "Adaptive Risk Management",
            "Live Market Data Integration"
        ]
    }


@app.get("/health", tags=["health"])
async def health_check(request: Request):
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": str(asyncio.get_event_loop().time()),
        "version": settings.version,
        "environment": settings.environment
    }
    
    # Check database connectivity
    try:
        if hasattr(request.app.state, 'db_manager') and request.app.state.db_manager:
            await request.app.state.db_manager.health_check()
            health_status["database"] = "connected"
        else:
            health_status["database"] = "not_initialized"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis connectivity
    try:
        if hasattr(request.app.state, 'redis_manager') and request.app.state.redis_manager:
            await request.app.state.redis_manager.health_check()
            health_status["redis"] = "connected"
        else:
            health_status["redis"] = "not_initialized"
    except Exception as e:
        health_status["redis"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check agent orchestrator
    try:
        if hasattr(request.app.state, 'agent_orchestrator') and request.app.state.agent_orchestrator.is_running:
            health_status["agents"] = "running"
        else:
            health_status["agents"] = "not_running_or_not_initialized"
    except Exception as e:
        health_status["agents"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check live data service
    try:
        if hasattr(request.app.state, 'live_data_service') and request.app.state.live_data_service.running:
            stats = request.app.state.live_data_service.get_statistics()
            health_status["live_data"] = {
                "status": "running",
                "quotes_fetched": stats["quotes_fetched"],
                "errors": stats["errors"],
                "uptime": stats["uptime"]
            }
        else:
            health_status["live_data"] = "not_running"
    except Exception as e:
        health_status["live_data"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
        
    if health_status["status"] == "degraded":
        return JSONResponse(status_code=503, content=health_status)
    
    return health_status


@app.get("/metrics/performance", tags=["Monitoring"])
@limiter.limit("10/minute")
async def get_performance_metrics(request: Request):
    """Get real-time performance metrics from the agent orchestrator"""
    if hasattr(request.app.state, 'agent_orchestrator'):
        return request.app.state.agent_orchestrator.get_performance_metrics()
    raise HTTPException(status_code=503, detail="Agent orchestrator not available")


@app.get("/metrics/live-data", tags=["Monitoring"])
@limiter.limit("10/minute")
async def get_live_data_metrics(request: Request):
    """Get live data service statistics"""
    if hasattr(request.app.state, 'live_data_service'):
        return request.app.state.live_data_service.get_statistics()
    raise HTTPException(status_code=503, detail="Live data service not available")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch unhandled errors"""
    logger.error(f"Unhandled exception for {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


# Dependency injectors
async def get_signal_service(request: Request) -> SignalService:
    if hasattr(request.app.state, 'signal_service'):
        return request.app.state.signal_service
    raise HTTPException(status_code=503, detail="Signal service not available")

async def get_market_data_service(request: Request) -> MarketDataService:
    if hasattr(request.app.state, 'market_data_service'):
        return request.app.state.market_data_service
    raise HTTPException(status_code=503, detail="Market data service not available")

async def get_live_data_service(request: Request) -> LiveDataService:
    if hasattr(request.app.state, 'live_data_service'):
        return request.app.state.live_data_service
    raise HTTPException(status_code=503, detail="Live data service not available")

async def get_agent_orchestrator(request: Request) -> AgentOrchestrator:
    if hasattr(request.app.state, 'agent_orchestrator'):
        return request.app.state.agent_orchestrator
    raise HTTPException(status_code=503, detail="Agent orchestrator not available")

async def get_db_manager(request: Request) -> DatabaseManager:
    if hasattr(request.app.state, 'db_manager'):
        return request.app.state.db_manager
    raise HTTPException(status_code=503, detail="Database manager not available")

async def get_redis_manager(request: Request) -> RedisManager:
    if hasattr(request.app.state, 'redis_manager'):
        return request.app.state.redis_manager
    raise HTTPException(status_code=503, detail="Redis manager not available")


# Data Models
class MarketData(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: int

class AISignal(BaseModel):
    id: str
    symbol: str
    type: str  # 'CALL' or 'PUT'
    strike: float
    expiry: str
    confidence: float
    entryPrice: float
    targetPrice: float
    stopLoss: float
    timeframe: str
    reasoning: str
    patterns: List[str]
    urgency: str  # 'HIGH', 'MEDIUM', 'LOW'

class AIInsight(BaseModel):
    levels: List[dict]
    signals: List[dict]
    trendLines: List[dict]
    analysis: dict


# Market Data Endpoints with rate limiting
@app.get("/api/v1/market-data/{symbol}")
@cache_response(expire_time_seconds=5)
@limiter.limit("10/minute")
async def get_market_data(request: Request, symbol: str):
    """Get real-time market data from live data service"""
    try:
        # Check if we're in test mode (use mock data)
        if os.getenv("TEST_MODE") == "true" or settings.environment == "test":
            from src.services.market_data_service_mock import MockMarketDataService
            mock_service = MockMarketDataService()
            tick, error = mock_service.fetch_real_time_data(symbol.upper())
            
            if error:
                if error.reason.value == "INVALID_SYMBOL":
                    raise HTTPException(status_code=404, detail=error.message)
                else:
                    raise HTTPException(status_code=503, detail=error.message)
            
            if tick:
                return {
                    "symbol": tick.symbol,
                    "price": tick.price,
                    "change": tick.change,
                    "change_percent": tick.change_percent,
                    "volume": tick.volume,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "high": tick.price * 1.02,  # Mock high
                    "low": tick.price * 0.98,   # Mock low
                    "timestamp": int(datetime.now().timestamp())
                }
        
        # Normal mode - use live data service
        live_data_service = request.app.state.live_data_service
        quote = await live_data_service.get_quote(symbol.upper())
        
        if not quote:
            raise HTTPException(status_code=404, detail="Symbol not found")
        
        return {
            "symbol": quote.symbol,
            "price": quote.price,
            "change": quote.price - quote.open,
            "change_percent": ((quote.price - quote.open) / quote.open * 100) if quote.open else 0,
            "volume": quote.volume,
            "bid": quote.bid,
            "ask": quote.ask,
            "high": quote.high,
            "low": quote.low,
            "timestamp": int(quote.timestamp.timestamp())
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market-data/{symbol}/historical")
@cache_response(expire_time_seconds=300)
@limiter.limit("30/minute")
async def get_historical_market_data(request: Request, symbol: str, period: str = "1D", interval: str = "5m"):
    """Get historical market data for a symbol"""
    try:
        # Check if we're in test mode (use mock data)
        if os.getenv("TEST_MODE") == "true" or settings.environment == "test":
            from src.services.market_data_service_mock import MockMarketDataService
            mock_service = MockMarketDataService()
            hist_data, error = mock_service.get_historical_data(symbol.upper(), period)
            
            if error:
                raise HTTPException(status_code=404, detail=error.message)
            
            # Transform pandas DataFrame to required format
            data = []
            if not hist_data.empty:
                for index, row in hist_data.iterrows():
                    data.append({
                        "timestamp": int(index.timestamp() * 1000),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"])
                    })
            
            return {"data": data}
        
        # Normal mode - use yfinance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        # Transform data to required format
        data = []
        for index, row in hist.iterrows():
            data.append({
                "timestamp": int(index.timestamp() * 1000),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })
        
        return {"data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch historical data: {str(e)}"
        )

@app.get("/api/v1/market-data/{symbol}/options")
@cache_response(expire_time_seconds=60)
@limiter.limit("20/minute")
async def get_options_chain(request: Request, symbol: str):
    """Get options chain data from live data service"""
    try:
        live_data_service = request.app.state.live_data_service
        options = await live_data_service.get_options_chain(symbol.upper())
        
        if not options:
            raise HTTPException(status_code=404, detail="No options data available")
        
        # Group by expiration and strike
        chain = {}
        for opt in options:
            exp = opt.expiration
            if exp not in chain:
                chain[exp] = {"calls": [], "puts": []}
            
            opt_data = {
                "strike": opt.strike,
                "bid": opt.bid,
                "ask": opt.ask,
                "last": opt.last,
                "volume": opt.volume,
                "openInterest": opt.open_interest,
                "impliedVolatility": opt.implied_volatility
            }
            
            if opt.option_type == "call":
                chain[exp]["calls"].append(opt_data)
            else:
                chain[exp]["puts"].append(opt_data)
        
        return {"symbol": symbol.upper(), "chain": chain}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching options data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AI Signals Endpoints with rate limiting
@app.get("/api/v1/signals/active")
@cache_response(expire_time_seconds=30)
@limiter.limit("30/minute")
async def get_active_signals(request: Request):
    """Get active trading signals from the orchestrator"""
    try:
        orchestrator = request.app.state.agent_orchestrator
        signals = await orchestrator.get_active_signals()
        
        # Transform to API format
        formatted_signals = []
        for signal in signals:
            formatted_signals.append({
                "id": signal.get("id", f"SIG{datetime.now().timestamp()}"),
                "symbol": signal["symbol"],
                "type": signal.get("option_type", "CALL").upper(),
                "strike": signal.get("strike", 0),
                "expiry": signal.get("expiry", ""),
                "confidence": signal["confidence"],
                "entryPrice": signal.get("entry_price", signal.get("current_price", 0)),
                "targetPrice": signal.get("target_price", 0),
                "stopLoss": signal.get("stop_loss", 0),
                "timeframe": signal.get("timeframe", "1 week"),
                "reasoning": signal.get("reasoning", ""),
                "patterns": signal.get("patterns", []),
                "urgency": "HIGH" if signal["confidence"] > 0.8 else "MEDIUM"
            })
        
        return formatted_signals
    except Exception as e:
        logger.error(f"Error fetching active signals: {e}")
        return []

@app.get("/api/v1/ai/insights/{symbol}")
@cache_response(expire_time_seconds=30)
@limiter.limit("30/minute")
async def get_ai_insights(request: Request, symbol: str):
    try:
        # Get data from live service
        live_data = request.app.state.live_data_service
        quote = await live_data.get_quote(symbol.upper())
        
        if not quote:
            raise HTTPException(status_code=404, detail="Symbol not found")
        
        # Generate AI insights
        current_price = quote.price
        
        # Calculate support/resistance levels
        levels = []
        for i in range(3):
            support = current_price * (1 - 0.02 * (i + 1))
            resistance = current_price * (1 + 0.02 * (i + 1))
            levels.append({
                "type": "support",
                "price": round(support, 2),
                "strength": 0.9 - (i * 0.2)
            })
            levels.append({
                "type": "resistance", 
                "price": round(resistance, 2),
                "strength": 0.9 - (i * 0.2)
            })
        
        # Generate signals from orchestrator
        orchestrator = request.app.state.agent_orchestrator
        agent_signals = orchestrator.generate_signals_for_symbol(symbol.upper())
        
        signals = []
        if agent_signals:
            signals.append({
                "type": agent_signals.get("action", "HOLD"),
                "confidence": agent_signals.get("confidence", 0.5),
                "price": current_price,
                "timestamp": datetime.now().isoformat()
            })
        
        # Create trend lines
        trend_lines = [
            {
                "type": "support",
                "start": {"x": 0, "y": current_price * 0.95},
                "end": {"x": 100, "y": current_price * 0.97},
                "strength": 0.7
            },
            {
                "type": "resistance",
                "start": {"x": 0, "y": current_price * 1.03},
                "end": {"x": 100, "y": current_price * 1.05},
                "strength": 0.8
            }
        ]
        
        # Analysis summary
        analysis = {
            "trend": "bullish" if quote.price > quote.open else "bearish",
            "momentum": "strong" if abs(quote.price - quote.open) / quote.open > 0.02 else "weak",
            "volume": "high" if quote.volume > 1000000 else "normal",
            "volatility": "high" if abs(quote.high - quote.low) / quote.price > 0.03 else "normal",
            "recommendation": agent_signals.get("action", "HOLD") if agent_signals else "HOLD"
        }
        
        return {
            "levels": levels,
            "signals": signals,
            "trendLines": trend_lines,
            "analysis": analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating AI insights for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    ) 