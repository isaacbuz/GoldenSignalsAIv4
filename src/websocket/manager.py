"""
WebSocket Manager - GoldenSignalsAI V3

Real-time WebSocket connection management, broadcasting, and event handling.
Supports multiple connection types, authentication, and message routing.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from src.core.redis_manager import RedisManager


class ConnectionInfo:
    """Information about a WebSocket connection"""
    
    def __init__(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str] = None,
        connection_type: str = "general",
        subscriptions: Optional[Set[str]] = None
    ):
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_id = user_id
        self.connection_type = connection_type
        self.subscriptions = subscriptions or set()
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.message_count = 0


class WebSocketManager:
    """
    Comprehensive WebSocket manager for real-time communication
    """
    
    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self.symbol_subscribers: Dict[str, Set[str]] = {}  # symbol -> connection_ids
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self._pubsub_task: Optional[asyncio.Task] = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the WebSocket manager"""
        try:
            # Start Redis pub/sub listener
            self._pubsub_task = asyncio.create_task(self._redis_pubsub_listener())
            self._initialized = True
            logger.info("WebSocketManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocketManager: {str(e)}")
            raise
    
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str] = None,
        connection_type: str = "general"
    ) -> bool:
        """
        Accept and manage a new WebSocket connection
        
        Args:
            websocket: FastAPI WebSocket instance
            connection_id: Unique connection identifier
            user_id: Optional authenticated user ID
            connection_type: Type of connection (signals, market_data, etc.)
            
        Returns:
            bool: Success status
        """
        try:
            await websocket.accept()
            
            # Create connection info
            connection_info = ConnectionInfo(
                websocket=websocket,
                connection_id=connection_id,
                user_id=user_id,
                connection_type=connection_type
            )
            
            # Store connection
            self.active_connections[connection_id] = connection_info
            
            # Track user connections
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
            
            # Send welcome message
            await self.send_to_connection(connection_id, {
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat(),
                "connection_type": connection_type
            })
            
            logger.info(f"WebSocket connection established: {connection_id} (user: {user_id}, type: {connection_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection {connection_id}: {str(e)}")
            return False
    
    async def disconnect(self, connection_id: str) -> None:
        """
        Handle WebSocket disconnection and cleanup
        
        Args:
            connection_id: Connection identifier to disconnect
        """
        try:
            connection_info = self.active_connections.get(connection_id)
            if not connection_info:
                return
            
            # Remove from symbol subscriptions
            for symbol in connection_info.subscriptions:
                if symbol in self.symbol_subscribers:
                    self.symbol_subscribers[symbol].discard(connection_id)
                    if not self.symbol_subscribers[symbol]:
                        del self.symbol_subscribers[symbol]
            
            # Remove from user connections
            if connection_info.user_id:
                if connection_info.user_id in self.user_connections:
                    self.user_connections[connection_info.user_id].discard(connection_id)
                    if not self.user_connections[connection_info.user_id]:
                        del self.user_connections[connection_info.user_id]
            
            # Remove from active connections
            del self.active_connections[connection_id]
            
            logger.info(f"WebSocket connection disconnected: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error during WebSocket disconnection {connection_id}: {str(e)}")
    
    async def send_to_connection(
        self,
        connection_id: str,
        message: Union[Dict[str, Any], str]
    ) -> bool:
        """
        Send message to a specific connection
        
        Args:
            connection_id: Target connection ID
            message: Message to send (dict or string)
            
        Returns:
            bool: Success status
        """
        try:
            connection_info = self.active_connections.get(connection_id)
            if not connection_info:
                logger.warning(f"Connection {connection_id} not found")
                return False
            
            # Prepare message
            if isinstance(message, dict):
                message_text = json.dumps(message)
            else:
                message_text = str(message)
            
            # Send message
            await connection_info.websocket.send_text(message_text)
            
            # Update activity tracking
            connection_info.last_activity = datetime.utcnow()
            connection_info.message_count += 1
            
            return True
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket {connection_id} disconnected during send")
            await self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {str(e)}")
            return False
    
    async def broadcast_to_all(
        self,
        message: Union[Dict[str, Any], str],
        connection_type: Optional[str] = None
    ) -> int:
        """
        Broadcast message to all connections or specific type
        
        Args:
            message: Message to broadcast
            connection_type: Optional filter by connection type
            
        Returns:
            int: Number of successful sends
        """
        successful_sends = 0
        
        for connection_id, connection_info in self.active_connections.items():
            # Filter by connection type if specified
            if connection_type and connection_info.connection_type != connection_type:
                continue
            
            success = await self.send_to_connection(connection_id, message)
            if success:
                successful_sends += 1
        
        return successful_sends
    
    async def broadcast_to_symbol_subscribers(
        self,
        symbol: str,
        message: Union[Dict[str, Any], str]
    ) -> int:
        """
        Broadcast message to all subscribers of a symbol
        
        Args:
            symbol: Stock symbol
            message: Message to broadcast
            
        Returns:
            int: Number of successful sends
        """
        symbol = symbol.upper()
        successful_sends = 0
        
        if symbol in self.symbol_subscribers:
            for connection_id in self.symbol_subscribers[symbol].copy():
                success = await self.send_to_connection(connection_id, message)
                if success:
                    successful_sends += 1
        
        return successful_sends
    
    async def broadcast_to_user(
        self,
        user_id: str,
        message: Union[Dict[str, Any], str]
    ) -> int:
        """
        Broadcast message to all connections of a user
        
        Args:
            user_id: User identifier
            message: Message to broadcast
            
        Returns:
            int: Number of successful sends
        """
        successful_sends = 0
        
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id].copy():
                success = await self.send_to_connection(connection_id, message)
                if success:
                    successful_sends += 1
        
        return successful_sends
    
    async def subscribe_to_symbol(
        self,
        connection_id: str,
        symbol: str
    ) -> bool:
        """
        Subscribe a connection to symbol updates
        
        Args:
            connection_id: Connection to subscribe
            symbol: Stock symbol to subscribe to
            
        Returns:
            bool: Success status
        """
        try:
            symbol = symbol.upper()
            connection_info = self.active_connections.get(connection_id)
            
            if not connection_info:
                logger.warning(f"Connection {connection_id} not found for symbol subscription")
                return False
            
            # Add to symbol subscribers
            if symbol not in self.symbol_subscribers:
                self.symbol_subscribers[symbol] = set()
            self.symbol_subscribers[symbol].add(connection_id)
            
            # Add to connection subscriptions
            connection_info.subscriptions.add(symbol)
            
            # Notify Redis to start streaming this symbol if needed
            await self.redis_manager.add_symbol_subscription(symbol)
            
            # Send confirmation
            await self.send_to_connection(connection_id, {
                "type": "subscription_confirmed",
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Connection {connection_id} subscribed to {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe {connection_id} to {symbol}: {str(e)}")
            return False
    
    async def unsubscribe_from_symbol(
        self,
        connection_id: str,
        symbol: str
    ) -> bool:
        """
        Unsubscribe a connection from symbol updates
        
        Args:
            connection_id: Connection to unsubscribe
            symbol: Stock symbol to unsubscribe from
            
        Returns:
            bool: Success status
        """
        try:
            symbol = symbol.upper()
            connection_info = self.active_connections.get(connection_id)
            
            if not connection_info:
                return False
            
            # Remove from symbol subscribers
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(connection_id)
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]
                    # Notify Redis to stop streaming this symbol
                    await self.redis_manager.remove_symbol_subscription(symbol)
            
            # Remove from connection subscriptions
            connection_info.subscriptions.discard(symbol)
            
            # Send confirmation
            await self.send_to_connection(connection_id, {
                "type": "unsubscription_confirmed",
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Connection {connection_id} unsubscribed from {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe {connection_id} from {symbol}: {str(e)}")
            return False
    
    async def handle_message(
        self,
        connection_id: str,
        message: str
    ) -> None:
        """
        Handle incoming WebSocket message
        
        Args:
            connection_id: Source connection ID
            message: Received message
        """
        try:
            # Parse JSON message
            data = json.loads(message)
            message_type = data.get("type")
            
            # Route message based on type
            if message_type == "subscribe":
                symbol = data.get("symbol")
                if symbol:
                    await self.subscribe_to_symbol(connection_id, symbol)
                    
            elif message_type == "unsubscribe":
                symbol = data.get("symbol")
                if symbol:
                    await self.unsubscribe_from_symbol(connection_id, symbol)
                    
            elif message_type == "ping":
                await self.send_to_connection(connection_id, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            elif message_type == "get_subscriptions":
                connection_info = self.active_connections.get(connection_id)
                if connection_info:
                    await self.send_to_connection(connection_id, {
                        "type": "subscriptions_list",
                        "subscriptions": list(connection_info.subscriptions),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            else:
                logger.warning(f"Unknown message type from {connection_id}: {message_type}")
                await self.send_to_connection(connection_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {connection_id}: {message}")
            await self.send_to_connection(connection_id, {
                "type": "error",
                "message": "Invalid JSON format",
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {str(e)}")
            await self.send_to_connection(connection_id, {
                "type": "error",
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _redis_pubsub_listener(self) -> None:
        """
        Listen for Redis pub/sub messages and broadcast to WebSocket clients
        """
        try:
            pubsub = self.redis_manager.redis.pubsub()
            
            # Subscribe to relevant channels
            await pubsub.subscribe("market_data:*", "signals:*", "alerts:*")
            
            logger.info("Started Redis pub/sub listener for WebSocket broadcasting")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        channel = message["channel"].decode()
                        data = json.loads(message["data"].decode())
                        
                        # Route message based on channel
                        if channel.startswith("market_data:"):
                            symbol = channel.split(":")[-1]
                            await self.broadcast_to_symbol_subscribers(symbol, {
                                "type": "market_data",
                                "symbol": symbol,
                                "data": data,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            
                        elif channel.startswith("signals:"):
                            symbol = channel.split(":")[-1]
                            await self.broadcast_to_symbol_subscribers(symbol, {
                                "type": "signal",
                                "symbol": symbol,
                                "data": data,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            
                        elif channel.startswith("alerts:"):
                            user_id = data.get("user_id")
                            if user_id:
                                await self.broadcast_to_user(user_id, {
                                    "type": "alert",
                                    "data": data,
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                            else:
                                await self.broadcast_to_all({
                                    "type": "alert",
                                    "data": data,
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                                
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Redis pub/sub listener error: {str(e)}")
        finally:
            await pubsub.close()
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get WebSocket connection statistics
        
        Returns:
            Dictionary with connection stats
        """
        try:
            # Count connections by type
            connection_types = {}
            total_messages = 0
            
            for connection_info in self.active_connections.values():
                conn_type = connection_info.connection_type
                connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
                total_messages += connection_info.message_count
            
            return {
                "total_connections": len(self.active_connections),
                "connection_types": connection_types,
                "symbol_subscriptions": len(self.symbol_subscribers),
                "user_connections": len(self.user_connections),
                "total_messages_sent": total_messages,
                "active_symbols": list(self.symbol_subscribers.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting connection stats: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup_stale_connections(self) -> int:
        """
        Clean up stale WebSocket connections
        
        Returns:
            Number of connections cleaned up
        """
        cleaned_up = 0
        current_time = datetime.utcnow()
        stale_threshold = 300  # 5 minutes
        
        stale_connections = []
        
        for connection_id, connection_info in self.active_connections.items():
            if (current_time - connection_info.last_activity).total_seconds() > stale_threshold:
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            try:
                await self.disconnect(connection_id)
                cleaned_up += 1
            except Exception as e:
                logger.error(f"Error cleaning up stale connection {connection_id}: {str(e)}")
        
        if cleaned_up > 0:
            logger.info(f"Cleaned up {cleaned_up} stale WebSocket connections")
        
        return cleaned_up
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the WebSocket manager"""
        try:
            self._initialized = False
            
            # Cancel Redis pub/sub task
            if self._pubsub_task:
                self._pubsub_task.cancel()
            
            # Close all connections
            for connection_id in list(self.active_connections.keys()):
                await self.disconnect(connection_id)
            
            logger.info("WebSocketManager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during WebSocketManager shutdown: {str(e)}") 