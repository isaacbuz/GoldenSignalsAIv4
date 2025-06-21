"""
WebSocket API for GoldenSignalsAI V3
Real-time data streaming and bidirectional communication
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Dict, Set, Optional, List, Any
import json
import asyncio
import logging
from datetime import datetime
from uuid import uuid4

from src.websocket.manager import WebSocketManager, ConnectionInfo
from src.services.market_data_service import MarketDataService
from agents.orchestration.simple_orchestrator import SimpleOrchestrator
from src.core.redis_manager import RedisManager
from src.core.auth import get_current_user_ws

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

# Global instances
ws_manager = WebSocketManager()
market_service = MarketDataService()
orchestrator = SimpleOrchestrator()
redis_manager = RedisManager()

# Active subscriptions tracking
active_subscriptions: Dict[str, Set[str]] = {
    "market_data": set(),
    "signals": set(),
    "agent_status": set(),
}

@router.websocket("")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    Main WebSocket endpoint for real-time data streaming
    
    Supports:
    - Market data streaming
    - Signal updates
    - Agent status monitoring
    - Portfolio updates
    - System alerts
    """
    connection_id = str(uuid4())
    user_id = None
    
    # Optional authentication
    if token:
        try:
            user = await get_current_user_ws(token)
            user_id = user.id if user else None
        except Exception as e:
            logger.warning(f"WebSocket auth failed: {e}")
    
    # Create connection info
    connection = ConnectionInfo(
        websocket=websocket,
        connection_id=connection_id,
        user_id=user_id,
        connection_type="general"
    )
    
    try:
        # Accept connection
        await ws_manager.connect(connection)
        logger.info(f"WebSocket connected: {connection_id}")
        
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "data": {
                "connection_id": connection_id,
                "status": "connected",
                "server_time": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Start background tasks
        tasks = [
            asyncio.create_task(handle_client_messages(connection)),
            asyncio.create_task(stream_market_data(connection)),
            asyncio.create_task(stream_signals(connection)),
            asyncio.create_task(stream_agent_status(connection)),
            asyncio.create_task(handle_redis_events(connection)),
        ]
        
        # Wait for disconnect
        await asyncio.gather(*tasks)
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        # Clean up
        await ws_manager.disconnect(connection_id)
        
        # Cancel background tasks
        for task in tasks:
            if not task.done():
                task.cancel()

async def handle_client_messages(connection: ConnectionInfo):
    """Handle incoming messages from client"""
    try:
        while True:
            data = await connection.websocket.receive_text()
            message = json.loads(data)
            
            await process_client_message(connection, message)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error handling client message: {e}")

async def process_client_message(connection: ConnectionInfo, message: Dict[str, Any]):
    """Process different types of client messages"""
    msg_type = message.get("type")
    data = message.get("data", {})
    
    if msg_type == "subscribe":
        await handle_subscription(connection, data, True)
    
    elif msg_type == "unsubscribe":
        await handle_subscription(connection, data, False)
    
    elif msg_type == "request":
        await handle_request(connection, data)
    
    elif msg_type == "heartbeat":
        # Echo heartbeat
        await connection.websocket.send_json({
            "type": "heartbeat",
            "data": {"server_time": datetime.utcnow().isoformat()},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    else:
        logger.warning(f"Unknown message type: {msg_type}")

async def handle_subscription(connection: ConnectionInfo, data: Dict[str, Any], subscribe: bool):
    """Handle subscription/unsubscription requests"""
    channel = data.get("channel")
    
    if channel == "market_data":
        symbol = data.get("symbol")
        if symbol:
            if subscribe:
                connection.subscriptions.add(f"market_data:{symbol}")
                active_subscriptions["market_data"].add(symbol)
                logger.info(f"Subscribed to market_data:{symbol}")
            else:
                connection.subscriptions.discard(f"market_data:{symbol}")
                # Check if any connection still needs this symbol
                if not any(f"market_data:{symbol}" in conn.subscriptions 
                          for conn in ws_manager.connections.values()):
                    active_subscriptions["market_data"].discard(symbol)
    
    elif channel == "signals":
        symbol = data.get("symbol")
        if symbol:
            if subscribe:
                connection.subscriptions.add(f"signals:{symbol}")
                active_subscriptions["signals"].add(symbol)
            else:
                connection.subscriptions.discard(f"signals:{symbol}")
    
    elif channel == "agent_status":
        agent = data.get("agent")
        if agent:
            if subscribe:
                connection.subscriptions.add(f"agent_status:{agent}")
                active_subscriptions["agent_status"].add(agent)
            else:
                connection.subscriptions.discard(f"agent_status:{agent}")

async def handle_request(connection: ConnectionInfo, data: Dict[str, Any]):
    """Handle one-time data requests"""
    action = data.get("action")
    
    if action == "market_data":
        symbols = data.get("symbols", [])
        for symbol in symbols:
            tick, error = market_service.fetch_real_time_data(symbol)
            if tick:
                await connection.websocket.send_json({
                    "type": "market_data",
                    "data": {
                        "symbol": tick.symbol,
                        "price": tick.price,
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "volume": tick.volume,
                        "change": tick.change,
                        "change_percent": tick.change_percent,
                    },
                    "timestamp": tick.timestamp
                })
    
    elif action == "agent_status":
        # Get agent performance metrics
        metrics = orchestrator.get_performance_metrics()
        await connection.websocket.send_json({
            "type": "agent_status",
            "data": metrics,
            "timestamp": datetime.utcnow().isoformat()
        })

async def stream_market_data(connection: ConnectionInfo):
    """Stream market data for subscribed symbols"""
    try:
        while True:
            # Get symbols this connection is subscribed to
            subscribed_symbols = {
                sub.split(":")[1] for sub in connection.subscriptions 
                if sub.startswith("market_data:")
            }
            
            if subscribed_symbols:
                for symbol in subscribed_symbols:
                    tick, error = market_service.fetch_real_time_data(symbol)
                    if tick:
                        await connection.websocket.send_json({
                            "type": "market_data",
                            "data": {
                                "symbol": tick.symbol,
                                "price": tick.price,
                                "bid": tick.bid,
                                "ask": tick.ask,
                                "volume": tick.volume,
                                "change": tick.change,
                                "change_percent": tick.change_percent,
                                "timestamp": tick.timestamp
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })
            
            # Stream every 5 seconds
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"Error streaming market data: {e}")

async def stream_signals(connection: ConnectionInfo):
    """Stream trading signals for subscribed symbols"""
    try:
        while True:
            # Get symbols this connection is subscribed to
            subscribed_symbols = {
                sub.split(":")[1] for sub in connection.subscriptions 
                if sub.startswith("signals:")
            }
            
            if subscribed_symbols:
                for symbol in subscribed_symbols:
                    # Generate signal
                    signal = orchestrator.generate_signals_for_symbol(symbol)
                    
                    if signal and signal.get("confidence", 0) > 0.6:  # Only send confident signals
                        await connection.websocket.send_json({
                            "type": "signal",
                            "data": {
                                "signal_id": str(uuid4()),
                                "symbol": signal["symbol"],
                                "signal_type": signal["action"],
                                "confidence": signal["confidence"],
                                "source": "orchestrator",
                                "reasoning": signal["metadata"]["reasoning"],
                                "agent_breakdown": signal["metadata"]["agent_breakdown"],
                                "timestamp": signal["timestamp"]
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })
            
            # Generate signals every 30 seconds
            await asyncio.sleep(30)
            
    except Exception as e:
        logger.error(f"Error streaming signals: {e}")

async def stream_agent_status(connection: ConnectionInfo):
    """Stream agent status updates"""
    try:
        while True:
            # Check if subscribed to any agent status
            has_agent_subs = any(sub.startswith("agent_status:") for sub in connection.subscriptions)
            
            if has_agent_subs:
                # Get performance metrics
                metrics = orchestrator.get_performance_metrics()
                
                # Send overall summary
                await connection.websocket.send_json({
                    "type": "agent_status",
                    "data": {
                        "summary": metrics["summary"],
                        "timestamp": metrics["timestamp"]
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Send individual agent updates
                for agent_name, agent_metrics in metrics["agents"].items():
                    if f"agent_status:{agent_name}" in connection.subscriptions:
                        await connection.websocket.send_json({
                            "type": "agent_status",
                            "data": {
                                "agent_name": agent_name,
                                "metrics": agent_metrics,
                                "status": "active" if agent_metrics["total_signals"] > 0 else "idle"
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })
            
            # Update every 60 seconds
            await asyncio.sleep(60)
            
    except Exception as e:
        logger.error(f"Error streaming agent status: {e}")

async def handle_redis_events(connection: ConnectionInfo):
    """Handle Redis pub/sub events"""
    try:
        # Subscribe to Redis channels
        pubsub = await redis_manager.redis_client.pubsub()
        await pubsub.subscribe("alerts:*", "portfolio:*")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                channel = message["channel"].decode()
                data = json.loads(message["data"].decode())
                
                # System alerts
                if channel.startswith("alerts:"):
                    await connection.websocket.send_json({
                        "type": "system_alert",
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Portfolio updates (if user is authenticated)
                elif channel.startswith("portfolio:") and connection.user_id:
                    if channel.endswith(connection.user_id):
                        await connection.websocket.send_json({
                            "type": "portfolio_update",
                            "data": data,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
    except Exception as e:
        logger.error(f"Error handling Redis events: {e}")

# Additional endpoints for specific data streams

@router.websocket("/market/{symbol}")
async def market_data_websocket(websocket: WebSocket, symbol: str):
    """Dedicated WebSocket for single symbol market data"""
    connection_id = str(uuid4())
    connection = ConnectionInfo(
        websocket=websocket,
        connection_id=connection_id,
        connection_type="market_data"
    )
    
    try:
        await ws_manager.connect(connection)
        connection.subscriptions.add(f"market_data:{symbol}")
        
        while True:
            tick, error = market_service.fetch_real_time_data(symbol)
            if tick:
                await websocket.send_json({
                    "symbol": tick.symbol,
                    "price": tick.price,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "volume": tick.volume,
                    "change": tick.change,
                    "change_percent": tick.change_percent,
                    "timestamp": tick.timestamp
                })
            
            await asyncio.sleep(1)  # 1 second updates
            
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(connection_id)

@router.websocket("/signals")
async def signals_websocket(websocket: WebSocket):
    """Dedicated WebSocket for all signals"""
    connection_id = str(uuid4())
    connection = ConnectionInfo(
        websocket=websocket,
        connection_id=connection_id,
        connection_type="signals"
    )
    
    try:
        await ws_manager.connect(connection)
        
        # Subscribe to all symbols
        symbols = orchestrator.symbols
        for symbol in symbols:
            connection.subscriptions.add(f"signals:{symbol}")
        
        while True:
            # Generate signals for all symbols
            signals = orchestrator.generate_all_signals()
            
            for signal in signals:
                if signal.get("confidence", 0) > 0.6:
                    await websocket.send_json({
                        "signal_id": str(uuid4()),
                        "symbol": signal["symbol"],
                        "signal_type": signal["action"],
                        "confidence": signal["confidence"],
                        "reasoning": signal["metadata"]["reasoning"],
                        "timestamp": signal["timestamp"]
                    })
            
            await asyncio.sleep(30)  # 30 second updates
            
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(connection_id) 