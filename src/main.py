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
from typing import Any, Dict, List

import sentry_sdk
from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.sessions import SessionMiddleware

from .core.config import settings
from .core.database import DatabaseManager
from .core.redis_manager import RedisManager
from .core.logging_config import setup_logging
from .agents import AgentOrchestrator
from .api.v1 import api_router
from .websocket.manager import WebSocketManager
from .services.signal_service import SignalService
from .services.market_data_service import MarketDataService
from .middleware.security import SecurityMiddleware
from .middleware.monitoring import MonitoringMiddleware


# Initialize logging
logger = setup_logging()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global managers (will be initialized in lifespan)
db_manager: DatabaseManager = None
redis_manager: RedisManager = None
agent_orchestrator: AgentOrchestrator = None
websocket_manager: WebSocketManager = None
signal_service: SignalService = None
market_data_service: MarketDataService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management
    
    Handles startup and shutdown of all services, connections, and background tasks.
    """
    global db_manager, redis_manager, agent_orchestrator, websocket_manager
    global signal_service, market_data_service
    
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
        db_manager = DatabaseManager()
        await db_manager.initialize()
        logger.info("✅ Database connections established")
        
        # Initialize Redis for caching and real-time data
        redis_manager = RedisManager()
        await redis_manager.initialize()
        logger.info("✅ Redis connections established")
        
        # Initialize WebSocket manager for real-time communication
        websocket_manager = WebSocketManager(redis_manager)
        await websocket_manager.initialize()
        logger.info("✅ WebSocket manager initialized")
        
        # Initialize core services
        signal_service = SignalService(db_manager, redis_manager)
        market_data_service = MarketDataService(redis_manager)
        await signal_service.initialize()
        await market_data_service.initialize()
        logger.info("✅ Core services initialized")
        
        # Initialize agent orchestrator
        agent_orchestrator = AgentOrchestrator(
            signal_service=signal_service,
            market_data_service=market_data_service,
            websocket_manager=websocket_manager
        )
        await agent_orchestrator.initialize()
        logger.info("✅ Agent orchestrator initialized")
        
        # Start background tasks
        asyncio.create_task(market_data_service.start_data_feed())
        asyncio.create_task(agent_orchestrator.start_signal_generation())
        logger.info("✅ Background tasks started")
        
        logger.info("🎯 GoldenSignalsAI V3 is fully operational!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {str(e)}")
        raise
    
    finally:
        # Cleanup on shutdown
        logger.info("🛑 Shutting down GoldenSignalsAI V3...")
        
        if agent_orchestrator:
            await agent_orchestrator.shutdown()
            
        if market_data_service:
            await market_data_service.shutdown()
            
        if signal_service:
            await signal_service.shutdown()
            
        if websocket_manager:
            await websocket_manager.shutdown()
            
        if redis_manager:
            await redis_manager.close()
            
        if db_manager:
            await db_manager.close()
            
        logger.info("✅ Graceful shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="GoldenSignalsAI V3",
    description="Next-Generation AI Trading Platform with Advanced Agentic Architecture",
    version="3.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

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


@app.get("/", tags=["Root"])
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
            "Adaptive Risk Management"
        ]
    }


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": str(asyncio.get_event_loop().time()),
        "version": settings.version,
        "environment": settings.environment
    }
    
    # Check database connectivity
    try:
        if db_manager:
            await db_manager.health_check()
            health_status["database"] = "connected"
        else:
            health_status["database"] = "not_initialized"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis connectivity
    try:
        if redis_manager:
            await redis_manager.health_check()
            health_status["redis"] = "connected"
        else:
            health_status["redis"] = "not_initialized"
    except Exception as e:
        health_status["redis"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check agent orchestrator
    try:
        if agent_orchestrator and agent_orchestrator.is_running:
            health_status["agents"] = "running"
            health_status["active_agents"] = len(agent_orchestrator.get_active_agents())
        else:
            health_status["agents"] = "not_running"
    except Exception as e:
        health_status["agents"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@app.get("/metrics/performance", tags=["Monitoring"])
@limiter.limit("10/minute")
async def get_performance_metrics(request):
    """Get detailed performance metrics"""
    if not agent_orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent orchestrator not initialized"
        )
    
    return await agent_orchestrator.get_performance_metrics()


@app.websocket("/ws/signals/{symbol}")
async def websocket_signals_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time signal streaming"""
    if not websocket_manager:
        await websocket.close(code=1011, reason="WebSocket manager not initialized")
        return
    
    await websocket_manager.connect_signal_stream(websocket, symbol)


@app.websocket("/ws/market-data/{symbol}")
async def websocket_market_data_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data streaming"""
    if not websocket_manager:
        await websocket.close(code=1011, reason="WebSocket manager not initialized")
        return
    
    await websocket_manager.connect_market_data_stream(websocket, symbol)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with detailed logging"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )


# Dependency injection functions
async def get_signal_service() -> SignalService:
    """Dependency to get signal service instance"""
    if not signal_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Signal service not available"
        )
    return signal_service


async def get_market_data_service() -> MarketDataService:
    """Dependency to get market data service instance"""
    if not market_data_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data service not available"
        )
    return market_data_service


async def get_agent_orchestrator() -> AgentOrchestrator:
    """Dependency to get agent orchestrator instance"""
    if not agent_orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent orchestrator not available"
        )
    return agent_orchestrator


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting GoldenSignalsAI V3 on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers if not settings.debug else 1,
        reload=settings.debug,
        log_level=settings.monitoring.log_level.lower(),
        access_log=True
    ) 