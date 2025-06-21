"""
WebSocket Manager for handling real-time connections
"""

from typing import Dict, List, Set
from fastapi import WebSocket
import uuid
import json


class WebSocketManager:
    """
    Manages WebSocket connections for real-time communication
    """
    
    def __init__(self):
        # Store active connections
        self.active_connections: Dict[str, WebSocket] = {}
        # Track subscriptions
        self.subscriptions: Dict[str, Set[str]] = {}
        
    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection
        """
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        self.subscriptions[connection_id] = set()
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """
        Remove a WebSocket connection
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.subscriptions:
            del self.subscriptions[connection_id]
    
    async def send_personal_message(self, message: str, connection_id: str):
        """
        Send a message to a specific connection
        """
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(message)
    
    async def send_personal_json(self, data: dict, connection_id: str):
        """
        Send JSON data to a specific connection
        """
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            await websocket.send_json(data)
    
    async def broadcast(self, message: str):
        """
        Broadcast a message to all connected clients
        """
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except:
                # Connection might be closed
                await self.disconnect(connection_id)
    
    async def broadcast_json(self, data: dict):
        """
        Broadcast JSON data to all connected clients
        """
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(data)
            except:
                # Mark for disconnection
                disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for conn_id in disconnected:
            await self.disconnect(conn_id)
    
    async def broadcast_to_subscribers(self, topic: str, data: dict):
        """
        Broadcast to clients subscribed to a specific topic
        """
        disconnected = []
        for connection_id, subscriptions in self.subscriptions.items():
            if topic in subscriptions:
                try:
                    websocket = self.active_connections[connection_id]
                    await websocket.send_json(data)
                except:
                    disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for conn_id in disconnected:
            await self.disconnect(conn_id)
    
    def subscribe(self, connection_id: str, topic: str):
        """
        Subscribe a connection to a topic
        """
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].add(topic)
    
    def unsubscribe(self, connection_id: str, topic: str):
        """
        Unsubscribe a connection from a topic
        """
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].discard(topic)
    
    def get_connection_count(self) -> int:
        """
        Get the number of active connections
        """
        return len(self.active_connections)
    
    def get_subscriptions(self, connection_id: str) -> Set[str]:
        """
        Get subscriptions for a connection
        """
        return self.subscriptions.get(connection_id, set()) 