#!/usr/bin/env python3
"""
🔥 GoldenSignalsAI V3 - Enhanced FastAPI Application
Modern version with lifespan handlers and enhanced features
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

# Import our services
try:
    from services.market_data_service import MarketDataService
except ImportError:
    try:
        # Try absolute import
        import sys
        sys.path.append('../')
        from src.services.market_data_service import MarketDataService
    except ImportError:
        # Fallback if import fails
        MarketDataService = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global services
market_data_service: Optional[MarketDataService] = None
connected_websockets: set = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan management for FastAPI"""
    global market_data_service
    
    # Startup
    logger.info("🚀 Starting GoldenSignalsAI V3 (Enhanced Version)...")
    
    try:
        # Initialize market data service if available
        if MarketDataService:
            market_data_service = MarketDataService()
            logger.info("✅ Market data service initialized with ML models")
        else:
            logger.warning("⚠️ Market data service not available")
        
        logger.info("🎯 GoldenSignalsAI V3 startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down GoldenSignalsAI V3...")
    logger.info("✅ Graceful shutdown completed")

# Create FastAPI application with modern lifespan
app = FastAPI(
    title="GoldenSignalsAI V3 Enhanced",
    description="Next-Generation AI Trading Platform with Enhanced Features",
    version="3.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with system information"""
    return {
        "name": "GoldenSignalsAI V3 Enhanced",
        "version": "3.0.1",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Advanced AI Trading System",
            "Real-time Signal Generation", 
            "Live Market Data Integration",
            "ML Model Predictions",
            "Risk Assessment",
            "WebSocket Streaming",
            "Modern FastAPI Lifespan"
        ]
    }

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Enhanced health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.1",
        "services": {
            "api": "running",
            "market_data": "available" if market_data_service else "unavailable"
        }
    }
    
    # Check ML models status
    if market_data_service and hasattr(market_data_service, 'ml_models'):
        if market_data_service.ml_models.models:
            health_data["services"]["ml_models"] = f"loaded ({len(market_data_service.ml_models.models)} models)"
        else:
            health_data["services"]["ml_models"] = "not_loaded"
    else:
        health_data["services"]["ml_models"] = "unavailable"
    
    # Add WebSocket connections count
    health_data["websocket_connections"] = len(connected_websockets)
    
    return health_data

@app.get("/api/v1/signals/{symbol}", tags=["Signals"])
async def get_signal(symbol: str):
    """Get trading signal for a symbol"""
    if not market_data_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data service not available"
        )
    
    try:
        # Generate signal
        signal = market_data_service.generate_signal(symbol.upper())
        
        if not signal:
            return {
                "symbol": symbol.upper(),
                "signal": "HOLD",
                "confidence": 0.5,
                "message": "Insufficient data for signal generation",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "symbol": signal.symbol,
            "signal": signal.signal_type,
            "confidence": signal.confidence,
            "price_target": signal.price_target,
            "stop_loss": signal.stop_loss,
            "risk_score": signal.risk_score,
            "indicators": signal.indicators,
            "timestamp": signal.timestamp
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
        # Get market tick
        tick = market_data_service.fetch_real_time_data(symbol.upper())
        
        if not tick:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data available for symbol {symbol}"
            )
        
        return {
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

@app.get("/api/v1/models/status", tags=["ML Models"])
async def get_model_status():
    """Get ML model status and information"""
    if not market_data_service or not hasattr(market_data_service, 'ml_models'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not available"
        )
    
    ml_models = market_data_service.ml_models
    return {
        "models_loaded": len(ml_models.models),
        "available_models": list(ml_models.models.keys()),
        "scalers_loaded": len(ml_models.scalers),
        "model_directory": ml_models.model_dir,
        "timestamp": datetime.now().isoformat()
    }

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
        
        # Keep connection alive and send periodic updates
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
    logger.info("🔥 Starting GoldenSignalsAI V3 Enhanced FastAPI Server")
    
    uvicorn.run(
        "src.main_v2:app",
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 