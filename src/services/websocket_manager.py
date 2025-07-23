"""
Enhanced WebSocket Manager
Handles real-time signal broadcasting with rooms and subscriptions
"""

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""

    SIGNAL_UPDATE = "signal_update"
    PRICE_UPDATE = "price_update"
    AGENT_UPDATE = "agent_update"
    DECISION_UPDATE = "decision_update"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    ERROR = "error"


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client"""

    id: str
    websocket: WebSocket
    subscriptions: Set[str]
    connected_at: datetime
    last_heartbeat: datetime
    metadata: Dict[str, Any]


@dataclass
class SignalUpdate:
    """Real-time signal update"""

    symbol: str
    signal_id: str
    action: str
    confidence: float
    price: float
    agents_consensus: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class WebSocketManager:
    """
    Enhanced WebSocket manager with room-based subscriptions
    """

    def __init__(self):
        # Client management
        self.clients: Dict[str, WebSocketClient] = {}

        # Room subscriptions (symbol -> client_ids)
        self.rooms: Dict[str, Set[str]] = {}

        # Message queue for reliable delivery
        self.message_queue: asyncio.Queue = asyncio.Queue()

        # Metrics
        self.metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "errors": 0,
        }

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background tasks for heartbeat and cleanup"""
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._message_processor())

    async def connect(
        self, websocket: WebSocket, client_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Connect a new WebSocket client"""
        await websocket.accept()

        # Generate client ID
        client_id = str(uuid.uuid4())

        # Create client object
        client = WebSocketClient(
            id=client_id,
            websocket=websocket,
            subscriptions=set(),
            connected_at=datetime.now(),
            last_heartbeat=datetime.now(),
            metadata=client_metadata or {},
        )

        self.clients[client_id] = client
        self.metrics["total_connections"] += 1
        self.metrics["active_connections"] += 1

        # Send welcome message
        await self._send_to_client(
            client_id,
            {
                "type": MessageType.HEARTBEAT.value,
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Connected to GoldenSignals WebSocket",
            },
        )

        logger.info(f"Client {client_id} connected")
        return client_id

    async def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.clients:
            client = self.clients[client_id]

            # Remove from all rooms
            for symbol in list(client.subscriptions):
                await self.unsubscribe(client_id, symbol)

            # Remove client
            del self.clients[client_id]
            self.metrics["active_connections"] -= 1

            logger.info(f"Client {client_id} disconnected")

    async def subscribe(self, client_id: str, symbol: str):
        """Subscribe client to symbol updates"""
        if client_id not in self.clients:
            return

        client = self.clients[client_id]
        client.subscriptions.add(symbol)

        # Add to room
        if symbol not in self.rooms:
            self.rooms[symbol] = set()
        self.rooms[symbol].add(client_id)

        # Send confirmation
        await self._send_to_client(
            client_id,
            {
                "type": MessageType.SUBSCRIBE.value,
                "symbol": symbol,
                "status": "subscribed",
                "timestamp": datetime.now().isoformat(),
            },
        )

        logger.info(f"Client {client_id} subscribed to {symbol}")

    async def unsubscribe(self, client_id: str, symbol: str):
        """Unsubscribe client from symbol updates"""
        if client_id not in self.clients:
            return

        client = self.clients[client_id]
        client.subscriptions.discard(symbol)

        # Remove from room
        if symbol in self.rooms:
            self.rooms[symbol].discard(client_id)
            if not self.rooms[symbol]:
                del self.rooms[symbol]

        # Send confirmation
        await self._send_to_client(
            client_id,
            {
                "type": MessageType.UNSUBSCRIBE.value,
                "symbol": symbol,
                "status": "unsubscribed",
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def broadcast_signal(self, signal: SignalUpdate):
        """Broadcast signal update to subscribed clients"""
        message = {
            "type": MessageType.SIGNAL_UPDATE.value,
            "data": {
                "symbol": signal.symbol,
                "signal_id": signal.signal_id,
                "action": signal.action,
                "confidence": signal.confidence,
                "price": signal.price,
                "agents_consensus": signal.agents_consensus,
                "timestamp": signal.timestamp.isoformat(),
                "metadata": signal.metadata,
            },
        }

        await self._broadcast_to_room(signal.symbol, message)

    async def broadcast_price_update(self, symbol: str, price: float, volume: int):
        """Broadcast price update to subscribed clients"""
        message = {
            "type": MessageType.PRICE_UPDATE.value,
            "data": {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": datetime.now().isoformat(),
            },
        }

        await self._broadcast_to_room(symbol, message)

    async def broadcast_agent_update(
        self, symbol: str, agent_name: str, signal: str, confidence: float
    ):
        """Broadcast individual agent update"""
        message = {
            "type": MessageType.AGENT_UPDATE.value,
            "data": {
                "symbol": symbol,
                "agent": agent_name,
                "signal": signal,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
            },
        }

        await self._broadcast_to_room(symbol, message)

    async def broadcast_decision(self, symbol: str, decision: Dict[str, Any]):
        """Broadcast final trading decision"""
        message = {
            "type": MessageType.DECISION_UPDATE.value,
            "data": {
                "symbol": symbol,
                "decision": decision,
                "timestamp": datetime.now().isoformat(),
            },
        }

        await self._broadcast_to_room(symbol, message)

    async def send_alert(
        self, client_id: str, alert_type: str, message: str, severity: str = "info"
    ):
        """Send alert to specific client"""
        await self._send_to_client(
            client_id,
            {
                "type": MessageType.ALERT.value,
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def handle_message(self, client_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        try:
            message_type = data.get("type")

            if message_type == MessageType.SUBSCRIBE.value:
                symbol = data.get("symbol")
                if symbol:
                    await self.subscribe(client_id, symbol)

            elif message_type == MessageType.UNSUBSCRIBE.value:
                symbol = data.get("symbol")
                if symbol:
                    await self.unsubscribe(client_id, symbol)

            elif message_type == MessageType.HEARTBEAT.value:
                # Update heartbeat
                if client_id in self.clients:
                    self.clients[client_id].last_heartbeat = datetime.now()

                # Echo heartbeat
                await self._send_to_client(
                    client_id,
                    {"type": MessageType.HEARTBEAT.value, "timestamp": datetime.now().isoformat()},
                )

            else:
                logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._send_to_client(
                client_id,
                {
                    "type": MessageType.ERROR.value,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    async def _send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        if client_id not in self.clients:
            return

        client = self.clients[client_id]
        try:
            await client.websocket.send_json(message)
            self.metrics["messages_sent"] += 1
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
            self.metrics["errors"] += 1
            await self.disconnect(client_id)

    async def _broadcast_to_room(self, room: str, message: Dict[str, Any]):
        """Broadcast message to all clients in a room"""
        if room not in self.rooms:
            return

        # Get list of clients to avoid modification during iteration
        client_ids = list(self.rooms[room])

        # Send to all clients in parallel
        tasks = [self._send_to_client(client_id, message) for client_id in client_ids]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats and clean up dead connections"""
        while True:
            try:
                await asyncio.sleep(30)  # 30 second heartbeat

                current_time = datetime.now()
                disconnected = []

                for client_id, client in self.clients.items():
                    # Check if client is alive
                    time_since_heartbeat = (current_time - client.last_heartbeat).total_seconds()

                    if time_since_heartbeat > 60:  # 60 second timeout
                        disconnected.append(client_id)
                    else:
                        # Send heartbeat
                        await self._send_to_client(
                            client_id,
                            {
                                "type": MessageType.HEARTBEAT.value,
                                "timestamp": current_time.isoformat(),
                            },
                        )

                # Disconnect dead clients
                for client_id in disconnected:
                    await self.disconnect(client_id)

            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    async def _message_processor(self):
        """Process queued messages"""
        while True:
            try:
                # Process messages from queue
                message = await self.message_queue.get()

                # Handle different message types
                if message["type"] == "broadcast_signal":
                    await self.broadcast_signal(message["signal"])
                elif message["type"] == "broadcast_price":
                    await self.broadcast_price_update(
                        message["symbol"], message["price"], message["volume"]
                    )

            except Exception as e:
                logger.error(f"Message processor error: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get WebSocket metrics"""
        room_stats = {symbol: len(clients) for symbol, clients in self.rooms.items()}

        return {
            **self.metrics,
            "rooms": room_stats,
            "total_subscriptions": sum(
                len(client.subscriptions) for client in self.clients.values()
            ),
        }

    async def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific client"""
        if client_id not in self.clients:
            return None

        client = self.clients[client_id]
        return {
            "id": client.id,
            "subscriptions": list(client.subscriptions),
            "connected_at": client.connected_at.isoformat(),
            "last_heartbeat": client.last_heartbeat.isoformat(),
            "metadata": client.metadata,
        }

    # Backward compatibility methods
    async def broadcast_json(self, data: dict):
        """Broadcast JSON data to all connected clients (backward compatibility)"""
        for client_id in list(self.clients.keys()):
            await self._send_to_client(client_id, data)

    async def broadcast_to_subscribers(self, topic: str, data: dict):
        """Broadcast to clients subscribed to a specific topic (backward compatibility)"""
        await self._broadcast_to_room(topic, data)

    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.clients)

    # Legacy properties for compatibility
    @property
    def active_connections(self) -> Dict[str, WebSocket]:
        """Legacy property for backward compatibility"""
        return {client_id: client.websocket for client_id, client in self.clients.items()}

    @property
    def subscriptions(self) -> Dict[str, Set[str]]:
        """Legacy property for backward compatibility"""
        return {client_id: client.subscriptions for client_id, client in self.clients.items()}


# Singleton instance
ws_manager = WebSocketManager()


# Helper functions for easy integration
async def broadcast_new_signal(signal_data: Dict[str, Any]):
    """Broadcast a new signal to subscribers"""
    signal = SignalUpdate(
        symbol=signal_data["symbol"],
        signal_id=signal_data.get("id", str(uuid.uuid4())),
        action=signal_data["action"],
        confidence=signal_data["confidence"],
        price=signal_data["price"],
        agents_consensus=signal_data.get("consensus", {}),
        timestamp=datetime.now(),
        metadata=signal_data.get("metadata"),
    )

    await ws_manager.broadcast_signal(signal)


async def notify_price_change(symbol: str, price: float, volume: int):
    """Notify subscribers of price change"""
    await ws_manager.broadcast_price_update(symbol, price, volume)
