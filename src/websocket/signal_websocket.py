"""
Unified WebSocket Server for Signal Platform
Handles all real-time communication with optimized performance
"""

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class MessageType(Enum):
    SIGNAL = "signal"
    AGENT = "agent"
    CONSENSUS = "consensus"
    MODEL = "model"
    ALERT = "alert"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"


class WebSocketTopic(Enum):
    SIGNALS_LIVE = "signals/live"
    AGENTS_STATUS = "agents/status"
    CONSENSUS_UPDATES = "consensus/updates"
    MODELS_PERFORMANCE = "models/performance"
    ALERTS_USER = "alerts/user"


@dataclass
class WebSocketMessage:
    type: str
    action: str
    timestamp: str
    data: Any
    metadata: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class ConnectionManager:
    """Manages WebSocket connections and subscriptions"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        """Accept and register a new connection"""
        await websocket.accept()
        async with self._lock:
            self.active_connections[connection_id] = websocket
            self.connection_metadata[connection_id] = {
                "connected_at": datetime.utcnow().isoformat(),
                "subscriptions": set(),
                "last_ping": datetime.utcnow(),
            }
        logger.info(f"WebSocket connected: {connection_id}")

    async def disconnect(self, connection_id: str) -> None:
        """Remove a connection and its subscriptions"""
        async with self._lock:
            # Remove from all subscriptions
            for topic in list(self.subscriptions.keys()):
                self.subscriptions[topic].discard(connection_id)
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]

            # Remove connection
            self.active_connections.pop(connection_id, None)
            self.connection_metadata.pop(connection_id, None)
        logger.info(f"WebSocket disconnected: {connection_id}")

    async def subscribe(self, connection_id: str, topic: str) -> None:
        """Subscribe a connection to a topic"""
        async with self._lock:
            self.subscriptions[topic].add(connection_id)
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscriptions"].add(topic)
        logger.debug(f"Connection {connection_id} subscribed to {topic}")

    async def unsubscribe(self, connection_id: str, topic: str) -> None:
        """Unsubscribe a connection from a topic"""
        async with self._lock:
            self.subscriptions[topic].discard(connection_id)
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscriptions"].discard(topic)
        logger.debug(f"Connection {connection_id} unsubscribed from {topic}")

    async def send_personal_message(self, message: str, connection_id: str) -> None:
        """Send a message to a specific connection"""
        websocket = self.active_connections.get(connection_id)
        if websocket:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                await self.disconnect(connection_id)

    async def broadcast_to_topic(self, message: str, topic: str) -> None:
        """Broadcast a message to all connections subscribed to a topic"""
        disconnected = []

        for connection_id in self.subscriptions.get(topic, set()).copy():
            websocket = self.active_connections.get(connection_id)
            if websocket:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {connection_id}: {e}")
                    disconnected.append(connection_id)

        # Clean up disconnected clients
        for connection_id in disconnected:
            await self.disconnect(connection_id)

    async def broadcast_to_all(self, message: str) -> None:
        """Broadcast a message to all connected clients"""
        disconnected = []

        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)

        # Clean up disconnected clients
        for connection_id in disconnected:
            await self.disconnect(connection_id)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

    def get_subscription_stats(self) -> Dict[str, int]:
        """Get subscription statistics"""
        return {topic: len(subscribers) for topic, subscribers in self.subscriptions.items()}


class SignalWebSocketService:
    """Main WebSocket service for handling signal communications"""

    def __init__(self, redis_url: Optional[str] = None):
        self.manager = ConnectionManager()
        self.redis_url = redis_url or "redis://localhost:6379"
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the service and Redis connection"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established")

            # Start listening to Redis pub/sub
            self.pubsub_task = asyncio.create_task(self._redis_listener())
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    async def shutdown(self):
        """Cleanup resources"""
        if self.pubsub_task:
            self.pubsub_task.cancel()

        if self.redis_client:
            await self.redis_client.close()

    async def _redis_listener(self):
        """Listen to Redis pub/sub for cross-server communication"""
        if not self.redis_client:
            return

        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("signals:broadcast")

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    topic = data.get("topic", "signals/live")
                    await self.manager.broadcast_to_topic(json.dumps(data), topic)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Redis listener error: {e}")
        finally:
            await pubsub.unsubscribe("signals:broadcast")

    async def handle_connection(self, websocket: WebSocket):
        """Handle a WebSocket connection"""
        connection_id = str(uuid.uuid4())
        await self.manager.connect(websocket, connection_id)

        # Send welcome message
        welcome_msg = WebSocketMessage(
            type="system",
            action="connected",
            timestamp=datetime.utcnow().isoformat(),
            data={"connection_id": connection_id, "server_time": datetime.utcnow().isoformat()},
            metadata={"version": "1.0.0"},
        )
        await self.manager.send_personal_message(welcome_msg.to_json(), connection_id)

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                await self._handle_message(connection_id, message)

        except WebSocketDisconnect:
            await self.manager.disconnect(connection_id)
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
            await self.manager.disconnect(connection_id)

    async def _handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        msg_type = message.get("type", "").lower()

        if msg_type == "subscribe":
            topic = message.get("topic")
            if topic:
                await self.manager.subscribe(connection_id, topic)
                # Send confirmation
                confirm_msg = WebSocketMessage(
                    type="system",
                    action="subscribed",
                    timestamp=datetime.utcnow().isoformat(),
                    data={"topic": topic},
                )
                await self.manager.send_personal_message(confirm_msg.to_json(), connection_id)

        elif msg_type == "unsubscribe":
            topic = message.get("topic")
            if topic:
                await self.manager.unsubscribe(connection_id, topic)
                # Send confirmation
                confirm_msg = WebSocketMessage(
                    type="system",
                    action="unsubscribed",
                    timestamp=datetime.utcnow().isoformat(),
                    data={"topic": topic},
                )
                await self.manager.send_personal_message(confirm_msg.to_json(), connection_id)

        elif msg_type == "ping":
            # Respond with pong
            pong_msg = WebSocketMessage(
                type="pong",
                action="response",
                timestamp=datetime.utcnow().isoformat(),
                data={"client_time": message.get("timestamp")},
            )
            await self.manager.send_personal_message(pong_msg.to_json(), connection_id)

    async def broadcast_signal(self, signal_data: Dict[str, Any]):
        """Broadcast a new signal to all subscribers"""
        message = WebSocketMessage(
            type=MessageType.SIGNAL.value,
            action="create",
            timestamp=datetime.utcnow().isoformat(),
            data=signal_data,
            metadata={
                "confidence": signal_data.get("confidence"),
                "agents": signal_data.get("agents", []),
                "processing_time_ms": signal_data.get("processing_time_ms", 0),
            },
        )

        # Broadcast to WebSocket clients
        await self.manager.broadcast_to_topic(message.to_json(), WebSocketTopic.SIGNALS_LIVE.value)

        # Publish to Redis for cross-server communication
        if self.redis_client:
            await self.redis_client.publish(
                "signals:broadcast",
                json.dumps({"topic": WebSocketTopic.SIGNALS_LIVE.value, **asdict(message)}),
            )

    async def broadcast_agent_status(self, agent_data: Dict[str, Any]):
        """Broadcast agent status update"""
        message = WebSocketMessage(
            type=MessageType.AGENT.value,
            action="update",
            timestamp=datetime.utcnow().isoformat(),
            data=agent_data,
        )

        await self.manager.broadcast_to_topic(message.to_json(), WebSocketTopic.AGENTS_STATUS.value)

    async def broadcast_consensus(self, consensus_data: Dict[str, Any]):
        """Broadcast consensus update"""
        message = WebSocketMessage(
            type=MessageType.CONSENSUS.value,
            action="update",
            timestamp=datetime.utcnow().isoformat(),
            data=consensus_data,
            metadata={
                "confidence": consensus_data.get("confidence"),
                "agents": consensus_data.get("participating_agents", []),
                "processing_time_ms": consensus_data.get("processing_time_ms", 0),
            },
        )

        await self.manager.broadcast_to_topic(
            message.to_json(), WebSocketTopic.CONSENSUS_UPDATES.value
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket service statistics"""
        return {
            "connections": self.manager.get_connection_count(),
            "subscriptions": self.manager.get_subscription_stats(),
            "redis_connected": self.redis_client is not None,
        }


# Global instance
ws_service = SignalWebSocketService()


# FastAPI WebSocket endpoint
async def websocket_endpoint(websocket: WebSocket):
    """FastAPI WebSocket endpoint handler"""
    await ws_service.handle_connection(websocket)
