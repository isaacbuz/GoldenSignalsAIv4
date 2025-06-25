"""
FastAPI Integration for Scalable WebSocket
Issue #180: Real-Time WebSocket Scaling
Provides API endpoints and WebSocket routes for the scalable system
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from src.websocket.scalable_manager import ScalableWebSocketManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/ws/v2", tags=["websocket-v2"])

# Global manager instance (will be initialized by app)
ws_manager: Optional[ScalableWebSocketManager] = None


def get_ws_manager() -> ScalableWebSocketManager:
    """Dependency to get WebSocket manager"""
    if not ws_manager:
        raise HTTPException(status_code=500, detail="WebSocket manager not initialized")
    return ws_manager


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    client_id: Optional[str] = Query(None)
):
    """
    Main WebSocket endpoint for scalable connections
    
    Query params:
    - token: Authentication token (optional)
    - client_id: Client identifier (optional)
    """
    manager = get_ws_manager()
    connection_id = None
    
    try:
        # Extract user_id from token (mock for demo)
        user_id = None
        if token:
            # In production: validate token and extract user_id
            user_id = f"user_{token[:8]}"
        
        # Connect
        connection_id = await manager.connect(
            websocket=websocket,
            connection_id=client_id,
            user_id=user_id,
            metadata={"connected_at": datetime.utcnow().isoformat()}
        )
        
        # Handle messages
        while True:
            message = await websocket.receive_text()
            await manager.handle_message(connection_id, message)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if connection_id:
            await manager.disconnect(connection_id)


@router.websocket("/market/{symbol}")
async def market_data_websocket(
    websocket: WebSocket,
    symbol: str,
    token: Optional[str] = Query(None)
):
    """
    Dedicated WebSocket endpoint for market data streaming
    Auto-subscribes to the specified symbol
    """
    manager = get_ws_manager()
    connection_id = None
    
    try:
        # Connect
        connection_id = await manager.connect(
            websocket=websocket,
            metadata={"type": "market_data", "symbol": symbol}
        )
        
        # Auto-subscribe to symbol
        await manager.subscribe_to_symbol(connection_id, symbol)
        
        # Keep connection alive
        while True:
            # Wait for disconnect or heartbeat
            await asyncio.sleep(30)
            
    except WebSocketDisconnect:
        pass
    finally:
        if connection_id:
            await manager.disconnect(connection_id)


@router.websocket("/signals")
async def signals_websocket(
    websocket: WebSocket,
    symbols: str = Query(""),  # Comma-separated symbols
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for signal streaming
    Can subscribe to multiple symbols at once
    """
    manager = get_ws_manager()
    connection_id = None
    
    try:
        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        
        # Connect
        connection_id = await manager.connect(
            websocket=websocket,
            metadata={"type": "signals", "symbols": symbol_list}
        )
        
        # Auto-subscribe to symbols
        for symbol in symbol_list:
            await manager.subscribe_to_symbol(connection_id, symbol)
        
        # Handle messages
        while True:
            message = await websocket.receive_text()
            await manager.handle_message(connection_id, message)
            
    except WebSocketDisconnect:
        pass
    finally:
        if connection_id:
            await manager.disconnect(connection_id)


# REST API endpoints for WebSocket management

@router.get("/stats")
async def get_websocket_stats(manager: ScalableWebSocketManager = Depends(get_ws_manager)):
    """Get WebSocket server statistics"""
    stats = await manager.get_stats()
    return JSONResponse(content={
        "status": "healthy",
        "stats": stats,
        "timestamp": datetime.utcnow().isoformat()
    })


@router.post("/broadcast")
async def broadcast_message(
    message: Dict[str, Any],
    target_type: str = "all",
    target_id: Optional[str] = None,
    manager: ScalableWebSocketManager = Depends(get_ws_manager)
):
    """
    Broadcast message to WebSocket connections
    
    Args:
    - message: Message to broadcast
    - target_type: "all", "symbol", or "user"
    - target_id: Symbol or user_id (required for targeted broadcasts)
    """
    if target_type in ["symbol", "user"] and not target_id:
        raise HTTPException(status_code=400, detail=f"target_id required for {target_type} broadcast")
    
    count = await manager.broadcast_message(
        message=message,
        target_type=target_type,
        target_id=target_id
    )
    
    return {
        "success": True,
        "recipients": count,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/market-data/{symbol}")
async def send_market_data(
    symbol: str,
    data: Dict[str, Any],
    manager: ScalableWebSocketManager = Depends(get_ws_manager)
):
    """Send market data update to symbol subscribers"""
    await manager.send_market_data(symbol.upper(), data)
    
    return {
        "success": True,
        "symbol": symbol.upper(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/signal/{symbol}")
async def send_signal(
    symbol: str,
    signal_data: Dict[str, Any],
    manager: ScalableWebSocketManager = Depends(get_ws_manager)
):
    """Send trading signal to symbol subscribers"""
    await manager.send_signal(symbol.upper(), signal_data)
    
    return {
        "success": True,
        "symbol": symbol.upper(),
        "signal": signal_data,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/alert/{user_id}")
async def send_user_alert(
    user_id: str,
    alert_data: Dict[str, Any],
    manager: ScalableWebSocketManager = Depends(get_ws_manager)
):
    """Send alert to specific user"""
    await manager.send_alert(user_id, alert_data)
    
    return {
        "success": True,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat()
    }


# Integration with FastAPI app

def setup_scalable_websocket(app: FastAPI, redis_url: str):
    """
    Setup scalable WebSocket for FastAPI app
    Call this in your app startup
    """
    global ws_manager
    
    @app.on_event("startup")
    async def startup_websocket():
        """Initialize WebSocket manager on startup"""
        global ws_manager
        
        # Create manager with unique server ID
        import socket
        server_id = f"{socket.gethostname()}_{app.state.get('port', 8000)}"
        
        ws_manager = ScalableWebSocketManager(
            redis_url=redis_url,
            server_id=server_id
        )
        
        await ws_manager.initialize()
        
        # Store in app state for access from other parts
        app.state.ws_manager_v2 = ws_manager
        
        logger.info(f"Scalable WebSocket manager initialized: {server_id}")
    
    @app.on_event("shutdown")
    async def shutdown_websocket():
        """Cleanup WebSocket manager on shutdown"""
        if ws_manager:
            await ws_manager.shutdown()
            logger.info("Scalable WebSocket manager shutdown complete")
    
    # Include router
    app.include_router(router)


# Example usage in services

class MarketDataService:
    """Example service that pushes market data"""
    
    def __init__(self, ws_manager: ScalableWebSocketManager):
        self.ws_manager = ws_manager
    
    async def publish_price_update(self, symbol: str, price: float, volume: int):
        """Publish price update to subscribers"""
        await self.ws_manager.send_market_data(symbol, {
            "price": price,
            "volume": volume,
            "change": self._calculate_change(symbol, price),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def _calculate_change(self, symbol: str, price: float) -> float:
        # Mock calculation
        return 2.5


class SignalService:
    """Example service that publishes signals"""
    
    def __init__(self, ws_manager: ScalableWebSocketManager):
        self.ws_manager = ws_manager
    
    async def publish_signal(
        self, 
        symbol: str, 
        signal_type: str,
        confidence: float,
        metadata: Dict[str, Any]
    ):
        """Publish trading signal"""
        await self.ws_manager.send_signal(symbol, {
            "type": signal_type,
            "confidence": confidence,
            "metadata": metadata,
            "generated_at": datetime.utcnow().isoformat()
        })


class AlertService:
    """Example service for user alerts"""
    
    def __init__(self, ws_manager: ScalableWebSocketManager):
        self.ws_manager = ws_manager
    
    async def send_portfolio_alert(self, user_id: str, alert_type: str, details: Dict[str, Any]):
        """Send portfolio alert to user"""
        await self.ws_manager.send_alert(user_id, {
            "alert_type": alert_type,
            "details": details,
            "severity": self._determine_severity(alert_type),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def _determine_severity(self, alert_type: str) -> str:
        severity_map = {
            "stop_loss_triggered": "high",
            "target_reached": "medium",
            "position_update": "low"
        }
        return severity_map.get(alert_type, "low") 