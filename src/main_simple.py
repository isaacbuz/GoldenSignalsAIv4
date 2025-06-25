#!/usr/bin/env python3
"""
🔥 GoldenSignalsAI V3 - Simplified FastAPI Application
Working version without complex dependencies
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our services - bypass __init__.py to avoid dependency issues
try:
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Import directly from the module file to avoid __init__.py issues
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "market_data_service", 
        os.path.join(os.path.dirname(__file__), "services", "market_data_service.py")
    )
    market_data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(market_data_module)
    MarketDataService = market_data_module.MarketDataService
    
except Exception as e:
    print(f"⚠️ Could not import MarketDataService: {e}")
    MarketDataService = None

# Import the hybrid signals router
try:
    from src.api.v1.hybrid_signals import router as hybrid_router
    hybrid_available = True
except Exception as e:
    print(f"⚠️ Could not import hybrid router: {e}")
    hybrid_router = None
    hybrid_available = False

# Import the new signals router
from src.api.v1.signals import router as signals_router_v1
from src.api.v1.analytics import router as analytics_router_v1
from src.api.v1.agents import router as agents_router_v1
from src.api.v1.backtesting import router as backtesting_router_v1
from src.api.v1.strategies import router as strategies_router_v1
from src.api.v1.portfolio import router as portfolio_router_v1
from src.api.v1.integrated_signals import router as integrated_signals_router

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global services
market_data_service: Optional[MarketDataService] = None
connected_websockets: set = set()

# Create FastAPI application
app = FastAPI(
    title="GoldenSignalsAI V3",
    description="Next-Generation AI Trading Platform with Hybrid Sentiment System",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the hybrid router if available
if hybrid_router:
    app.include_router(hybrid_router)
    logger.info("✅ Hybrid sentiment system router included")

# API Routes
app.include_router(signals_router_v1)
app.include_router(analytics_router_v1)
app.include_router(agents_router_v1)
app.include_router(backtesting_router_v1)
app.include_router(strategies_router_v1)
app.include_router(portfolio_router_v1)
app.include_router(integrated_signals_router)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global market_data_service
    
    logger.info("🚀 Starting GoldenSignalsAI V3...")
    
    try:
        # Initialize market data service if available
        if MarketDataService:
            market_data_service = MarketDataService()
            # Update ML model path to correct location
            if hasattr(market_data_service, 'ml_models'):
                market_data_service.ml_models = market_data_service.ml_models.__class__(model_dir="../ml_training/models")
            logger.info("✅ Market data service initialized")
        else:
            logger.warning("⚠️ Market data service not available")
        
        logger.info("🎯 GoldenSignalsAI V3 startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down GoldenSignalsAI V3...")
    logger.info("✅ Graceful shutdown completed")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with system information"""
    return {
        "name": "GoldenSignalsAI V3",
        "version": "3.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Advanced AI Trading System",
            "Real-time Signal Generation", 
            "Live Market Data Integration",
            "ML Model Predictions",
            "Risk Assessment"
        ]
    }

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "services": {
            "api": "running",
            "market_data": "available" if market_data_service else "unavailable"
        }
    }

@app.get("/api/v1/signals/{symbol}", tags=["Signals"])
async def get_signal(symbol: str):
    """Get trading signal for a symbol"""
    if not market_data_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data service not available"
        )
    
    try:
        # Check market hours first
        market_hours = market_data_service.check_market_hours()
        
        # Generate signal
        signal = market_data_service.generate_signal(symbol.upper())
        
        if not signal:
            return {
                "symbol": symbol.upper(),
                "signal": "HOLD",
                "confidence": 0.5,
                "message": "Insufficient data for signal generation",
                "timestamp": datetime.now().isoformat(),
                "market_status": {
                    "is_open": market_hours.is_open,
                    "reason": market_hours.reason,
                    "next_open": market_hours.next_open.isoformat() if market_hours.next_open else None
                }
            }
        
        return {
            "symbol": signal.symbol,
            "signal": signal.signal_type,
            "confidence": signal.confidence,
            "price_target": signal.price_target,
            "stop_loss": signal.stop_loss,
            "risk_score": signal.risk_score,
            "indicators": signal.indicators,
            "timestamp": signal.timestamp,
            "is_after_hours": signal.is_after_hours,
            "market_status": {
                "is_open": market_hours.is_open,
                "reason": market_hours.reason,
                "next_open": market_hours.next_open.isoformat() if market_hours.next_open else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating signal: {str(e)}"
        )

@app.get("/api/v1/market-data/{symbol}", tags=["Market Data"])
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    if not market_data_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data service not available"
        )
    
    try:
        # Get market tick with error handling
        tick, error = market_data_service.fetch_real_time_data(symbol.upper())
        
        if not tick:
            # Check if we have error information
            if error:
                # Return appropriate status based on error type
                # Import the enum from the module
                from src.services.market_data_service import DataUnavailableReason
                
                if error.reason == DataUnavailableReason.INVALID_SYMBOL:
                    status_code = status.HTTP_404_NOT_FOUND
                elif error.reason == DataUnavailableReason.API_LIMIT:
                    status_code = status.HTTP_429_TOO_MANY_REQUESTS
                else:
                    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                
                raise HTTPException(
                    status_code=status_code,
                    detail={
                        "message": error.message,
                        "reason": error.reason.value,
                        "is_recoverable": error.is_recoverable,
                        "suggested_action": error.suggested_action,
                        "timestamp": error.timestamp
                    }
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No data available for symbol {symbol}"
                )
        
        response_data = {
            "symbol": tick.symbol,
            "price": tick.price,
            "volume": tick.volume,
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.spread,
            "change": tick.change,
            "change_percent": tick.change_percent,
            "timestamp": tick.timestamp
        }
        
        # Add error warning if data is from cache
        if error:
            response_data["data_source"] = "cache"
            response_data["warning"] = error.message
        else:
            response_data["data_source"] = "live"
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching market data: {str(e)}"
        )

@app.get("/api/v1/market-summary", tags=["Market Data"])
async def get_market_summary():
    """Get market summary for all tracked symbols"""
    if not market_data_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data service not available"
        )
    
    try:
        summary = market_data_service.get_market_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting market summary: {str(e)}"
        )

@app.get("/api/v1/symbols", tags=["Market Data"])
async def get_available_symbols():
    """Get list of available symbols"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'SPY', 'QQQ', 'IWM']
    return {
        "symbols": symbols,
        "count": len(symbols),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/market-data/{symbol}/historical", tags=["Market Data"])
async def get_historical_data(symbol: str, period: str = "1d", interval: str = "5m"):
    """Get historical data for a symbol (mock implementation)"""
    if not market_data_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data service not available"
        )
    
    try:
        # Get current market data for base price
        current_data, _ = market_data_service.fetch_real_time_data(symbol.upper())
        base_price = current_data.price if current_data else 150.0
        
        # Generate mock historical data
        import random
        from datetime import timedelta
        
        # Determine number of data points based on period
        periods = {
            '1d': 96,    # 15-minute intervals for 1 day
            '5d': 120,   # hourly for 5 days
            '1M': 22,    # daily for 1 month
            '3M': 66,    # daily for 3 months
            '6M': 132,   # daily for 6 months
            '1Y': 252,   # daily for 1 year
        }
        
        point_count = periods.get(period, 96)
        now = datetime.now()
        
        historical_data = []
        for i in range(point_count):
            timestamp = now - timedelta(minutes=15 * (point_count - i))
            
            # Generate realistic price movement
            trend = random.uniform(-0.02, 0.02)
            volatility = random.uniform(-0.01, 0.01)
            price = base_price * (1 + trend + volatility)
            
            historical_data.append({
                "timestamp": timestamp.isoformat(),
                "open": price * (1 + random.uniform(-0.005, 0.005)),
                "high": price * (1 + random.uniform(0, 0.01)),
                "low": price * (1 + random.uniform(-0.01, 0)),
                "close": price,
                "volume": random.randint(100000, 1000000)
            })
        
        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "data": historical_data,
            "count": len(historical_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting historical data: {str(e)}"
        )

@app.get("/api/v1/analytics", tags=["Analytics"])
async def get_analytics(timeRange: str = "6M"):
    """Get analytics data (mock implementation)"""
    try:
        # Mock analytics data
        analytics = {
            "timeRange": timeRange,
            "summary": {
                "totalSignals": 1250,
                "successRate": 78.5,
                "avgReturnPerSignal": 2.3,
                "totalReturn": 12.7
            },
            "performance": {
                "dailyReturns": [0.5, 1.2, -0.3, 2.1, 0.8, -0.5, 1.5],
                "cumulativeReturn": 12.7,
                "maxDrawdown": -5.2,
                "sharpeRatio": 1.85
            },
            "signalBreakdown": {
                "BUY": 650,
                "SELL": 400,
                "HOLD": 200
            },
            "topPerformingSymbols": [
                {"symbol": "NVDA", "return": 25.3, "signals": 45},
                {"symbol": "AAPL", "return": 18.7, "signals": 62},
                {"symbol": "TSLA", "return": 15.2, "signals": 38}
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting analytics: {str(e)}"
        )

@app.get("/api/v1/market-data/status", tags=["Market Data"])
async def get_market_status():
    """Get market status information"""
    try:
        from datetime import time
        
        now = datetime.now()
        market_open = time(9, 30)  # 9:30 AM
        market_close = time(16, 0)  # 4:00 PM
        current_time = now.time()
        
        # Check if it's a weekday (0=Monday, 6=Sunday)
        is_weekday = now.weekday() < 5
        is_market_hours = market_open <= current_time <= market_close
        is_open = is_weekday and is_market_hours
        
        return {
            "is_open": is_open,
            "current_time": now.isoformat(),
            "market_hours": {
                "open": "09:30",
                "close": "16:00"
            },
            "next_open": "Next weekday 09:30",
            "next_close": "Next weekday 16:00",
            "timezone": "EST",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting market status: {str(e)}"
        )

@app.get("/api/v1/agents/performance", tags=["Agents"])
async def get_agent_performance():
    """Get AI agent performance metrics"""
    try:
        # Mock agent performance data
        agents_performance = {
            "signal_agent": {
                "agent_name": "signal_agent",
                "total_signals": 1250,
                "correct_signals": 1025,
                "accuracy": 82.0,
                "avg_confidence": 0.78,
                "last_updated": datetime.now().isoformat(),
            },
            "risk_agent": {
                "agent_name": "risk_agent", 
                "total_signals": 890,
                "correct_signals": 756,
                "accuracy": 84.9,
                "avg_confidence": 0.82,
                "last_updated": datetime.now().isoformat(),
            },
            "market_agent": {
                "agent_name": "market_agent",
                "total_signals": 2100,
                "correct_signals": 1680,
                "accuracy": 80.0,
                "avg_confidence": 0.75,
                "last_updated": datetime.now().isoformat(),
            }
        }
        
        return {
            "agents": agents_performance,
            "summary": {
                "total_agents": len(agents_performance),
                "avg_accuracy": sum(agent["accuracy"] for agent in agents_performance.values()) / len(agents_performance),
                "total_signals": sum(agent["total_signals"] for agent in agents_performance.values()),
                "total_correct": sum(agent["correct_signals"] for agent in agents_performance.values()),
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting agent performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting agent performance: {str(e)}"
        )

@app.websocket("/ws/signals/{symbol}")
async def websocket_signals_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time signal streaming"""
    await websocket.accept()
    connected_websockets.add(websocket)
    
    try:
        logger.info(f"🔌 WebSocket connected for signals: {symbol}")
        
        # Send initial signal
        if market_data_service:
            signal = market_data_service.generate_signal(symbol.upper())
            if signal:
                await websocket.send_json({
                    "type": "signal",
                    "data": {
                        "symbol": signal.symbol,
                        "signal": signal.signal_type,
                        "confidence": signal.confidence,
                        "price_target": signal.price_target,
                        "stop_loss": signal.stop_loss,
                        "risk_score": signal.risk_score,
                        "timestamp": signal.timestamp
                    }
                })
        
        # Keep connection alive
        while True:
            await asyncio.sleep(30)  # Send updates every 30 seconds
            
            if market_data_service:
                signal = market_data_service.generate_signal(symbol.upper())
                if signal:
                    await websocket.send_json({
                        "type": "signal_update",
                        "data": {
                            "symbol": signal.symbol,
                            "signal": signal.signal_type,
                            "confidence": signal.confidence,
                            "timestamp": signal.timestamp
                        }
                    })
                    
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {e}")
    finally:
        connected_websockets.discard(websocket)
        logger.info(f"🔌 WebSocket disconnected for signals: {symbol}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

def main():
    """Main function to run the server"""
    logger.info("🔥 Starting GoldenSignalsAI V3 FastAPI Server")
    
    uvicorn.run(
        "src.main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 