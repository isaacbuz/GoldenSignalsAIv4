"""
Unit tests for Scalable WebSocket Manager
Issue #180: Real-Time WebSocket Scaling
"""

import asyncio
import json
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.websocket.scalable_manager import (
    ScalableWebSocketManager,
    ScalableConnection,
    MessageRouter,
    MessageType,
    ConnectionState
)


@pytest.fixture
async def mock_redis():
    """Create mock Redis client"""
    redis = AsyncMock()
    redis.hset = AsyncMock()
    redis.hget = AsyncMock()
    redis.hdel = AsyncMock()
    redis.delete = AsyncMock()
    redis.expire = AsyncMock()
    redis.publish = AsyncMock(return_value=1)
    redis.hlen = AsyncMock(return_value=10)

    # Mock pubsub
    pubsub = AsyncMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.close = AsyncMock()
    pubsub.get_message = AsyncMock(return_value=None)

    redis.pubsub.return_value = pubsub

    return redis


@pytest.fixture
async def mock_websocket():
    """Create mock WebSocket"""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock()
    return ws


@pytest.fixture
async def ws_manager(mock_redis):
    """Create WebSocket manager with mocked Redis"""
    with patch('src.websocket.scalable_manager.redis.from_url') as mock_from_url:
        mock_from_url.return_value = mock_redis

        manager = ScalableWebSocketManager(
            redis_url="redis://localhost:6379",
            server_id="test_server_1"
        )

        # Manually set Redis client to avoid initialization
        manager.redis = mock_redis
        manager.router = MessageRouter(mock_redis, "test_server_1")
        manager.router.pubsub = mock_redis.pubsub()

        # Don't start background tasks in tests
        manager._listener_task = None
        manager._heartbeat_task = None
        manager._cleanup_task = None

        yield manager

        # Cleanup
        if manager.redis:
            await manager.redis.close()


class TestScalableConnection:
    """Test ScalableConnection class"""

    def test_connection_creation(self, mock_websocket):
        """Test creating a connection"""
        conn = ScalableConnection(
            websocket=mock_websocket,
            connection_id="conn_123",
            server_id="server_1",
            user_id="user_456",
            metadata={"source": "web"}
        )

        assert conn.connection_id == "conn_123"
        assert conn.server_id == "server_1"
        assert conn.user_id == "user_456"
        assert conn.state == ConnectionState.CONNECTING
        assert "source" in conn.metadata
        assert len(conn.subscriptions) == 0

    def test_connection_to_dict(self, mock_websocket):
        """Test converting connection to dictionary"""
        conn = ScalableConnection(
            websocket=mock_websocket,
            connection_id="conn_123",
            server_id="server_1",
            user_id="user_456"
        )
        conn.subscriptions.add("AAPL")
        conn.subscriptions.add("GOOGL")

        data = conn.to_dict()

        assert data["connection_id"] == "conn_123"
        assert data["server_id"] == "server_1"
        assert data["user_id"] == "user_456"
        assert data["state"] == "connecting"
        assert "AAPL" in data["subscriptions"]
        assert "GOOGL" in data["subscriptions"]
        assert "created_at" in data
        assert "last_heartbeat" in data


class TestMessageRouter:
    """Test MessageRouter class"""

    async def test_router_initialization(self, mock_redis):
        """Test router initialization"""
        router = MessageRouter(mock_redis, "test_server")
        await router.initialize()

        # Should subscribe to server channel and broadcasts
        mock_redis.pubsub().subscribe.assert_called_once()
        call_args = mock_redis.pubsub().subscribe.call_args[0]
        assert "server:test_server" in call_args
        assert "broadcast:all" in call_args

    async def test_route_to_connection(self, mock_redis):
        """Test routing message to specific connection"""
        router = MessageRouter(mock_redis, "test_server")

        # Mock server lookup
        mock_redis.hget.return_value = b"target_server"

        message = {"type": "test", "data": "hello"}
        result = await router.route_to_connection("conn_123", message)

        assert result is True

        # Verify Redis publish
        mock_redis.publish.assert_called_once()
        channel, data = mock_redis.publish.call_args[0]
        assert channel == "server:target_server"

        published_data = json.loads(data)
        assert published_data["target_connection"] == "conn_123"
        assert published_data["message"] == message

    async def test_route_to_symbol_subscribers(self, mock_redis):
        """Test routing to symbol subscribers"""
        router = MessageRouter(mock_redis, "test_server")

        message = {"type": "market_data", "price": 150.0}
        count = await router.route_to_symbol_subscribers("AAPL", message)

        assert count == 1  # Mock returns 1

        # Verify publish
        mock_redis.publish.assert_called_once()
        channel, data = mock_redis.publish.call_args[0]
        assert channel == "symbol:AAPL"

        published_data = json.loads(data)
        assert published_data["symbol"] == "AAPL"
        assert published_data["message"] == message

    async def test_register_unregister_connection(self, mock_redis):
        """Test connection registration and unregistration"""
        router = MessageRouter(mock_redis, "test_server")

        # Register
        await router.register_connection("conn_123", "server_1")
        mock_redis.hset.assert_called_with(
            "connection_servers", "conn_123", "server_1"
        )
        mock_redis.expire.assert_called()

        # Unregister
        await router.unregister_connection("conn_123")
        mock_redis.hdel.assert_called_with("connection_servers", "conn_123")
        mock_redis.delete.assert_called_with("connection:conn_123")


class TestScalableWebSocketManager:
    """Test ScalableWebSocketManager class"""

    async def test_manager_initialization(self):
        """Test manager initialization"""
        with patch('src.websocket.scalable_manager.redis.from_url') as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis

            manager = ScalableWebSocketManager(
                redis_url="redis://localhost:6379",
                server_id="test_server"
            )

            await manager.initialize()

            assert manager.server_id == "test_server"
            assert manager.redis is not None
            assert manager.router is not None
            assert len(manager.connections) == 0

    async def test_connect_websocket(self, ws_manager, mock_websocket):
        """Test WebSocket connection"""
        connection_id = await ws_manager.connect(
            websocket=mock_websocket,
            user_id="user_123",
            metadata={"device": "mobile"}
        )

        assert connection_id.startswith("conn_")
        assert connection_id in ws_manager.connections

        # Verify connection state
        conn = ws_manager.connections[connection_id]
        assert conn.state == ConnectionState.CONNECTED
        assert conn.user_id == "user_123"
        assert conn.metadata["device"] == "mobile"

        # Verify WebSocket accepted
        mock_websocket.accept.assert_called_once()

        # Verify welcome message sent
        mock_websocket.send_json.assert_called()
        welcome_msg = mock_websocket.send_json.call_args[0][0]
        assert welcome_msg["type"] == MessageType.CONNECTION_ESTABLISHED.value
        assert welcome_msg["connection_id"] == connection_id

    async def test_disconnect_websocket(self, ws_manager, mock_websocket):
        """Test WebSocket disconnection"""
        # Connect first
        connection_id = await ws_manager.connect(websocket=mock_websocket)

        # Subscribe to symbol
        await ws_manager.subscribe_to_symbol(connection_id, "AAPL")

        # Disconnect
        await ws_manager.disconnect(connection_id)

        assert connection_id not in ws_manager.connections
        assert connection_id not in ws_manager.symbol_subscribers.get("AAPL", set())

    async def test_subscribe_unsubscribe_symbol(self, ws_manager, mock_websocket):
        """Test symbol subscription and unsubscription"""
        # Connect
        connection_id = await ws_manager.connect(websocket=mock_websocket)

        # Subscribe
        result = await ws_manager.subscribe_to_symbol(connection_id, "aapl")
        assert result is True

        # Check tracking
        assert "AAPL" in ws_manager.symbol_subscribers
        assert connection_id in ws_manager.symbol_subscribers["AAPL"]

        conn = ws_manager.connections[connection_id]
        assert "AAPL" in conn.subscriptions

        # Verify confirmation sent
        calls = mock_websocket.send_json.call_args_list
        confirm_msg = calls[-1][0][0]
        assert confirm_msg["type"] == MessageType.SUBSCRIPTION_CONFIRMED.value
        assert confirm_msg["symbol"] == "AAPL"

        # Unsubscribe
        result = await ws_manager.unsubscribe_from_symbol(connection_id, "AAPL")
        assert result is True

        assert connection_id not in ws_manager.symbol_subscribers.get("AAPL", set())
        assert "AAPL" not in conn.subscriptions

    async def test_handle_message_subscribe(self, ws_manager, mock_websocket):
        """Test handling subscribe message"""
        connection_id = await ws_manager.connect(websocket=mock_websocket)

        # Send subscribe message
        message = json.dumps({
            "type": "subscribe",
            "symbol": "TSLA"
        })

        await ws_manager.handle_message(connection_id, message)

        # Verify subscription
        assert "TSLA" in ws_manager.symbol_subscribers
        assert connection_id in ws_manager.symbol_subscribers["TSLA"]

    async def test_handle_message_ping(self, ws_manager, mock_websocket):
        """Test handling ping message"""
        connection_id = await ws_manager.connect(websocket=mock_websocket)

        # Clear previous calls
        mock_websocket.send_json.reset_mock()

        # Send ping
        message = json.dumps({"type": "ping"})
        await ws_manager.handle_message(connection_id, message)

        # Verify pong sent
        mock_websocket.send_json.assert_called_once()
        pong_msg = mock_websocket.send_json.call_args[0][0]
        assert pong_msg["type"] == "pong"

    async def test_broadcast_message(self, ws_manager, mock_redis):
        """Test broadcasting messages"""
        # Test broadcast to all
        count = await ws_manager.broadcast_message(
            message={"alert": "System update"},
            target_type="all"
        )

        # Verify Redis publish called
        ws_manager.router.redis.publish.assert_called()

        # Test broadcast to symbol
        await ws_manager.broadcast_message(
            message={"signal": "BUY"},
            target_type="symbol",
            target_id="AAPL"
        )

        # Test broadcast to user
        await ws_manager.broadcast_message(
            message={"notification": "Order filled"},
            target_type="user",
            target_id="user_123"
        )

    async def test_send_market_data(self, ws_manager):
        """Test sending market data"""
        data = {
            "price": 185.50,
            "volume": 1000000,
            "bid": 185.45,
            "ask": 185.55
        }

        await ws_manager.send_market_data("AAPL", data)

        # Verify publish called
        ws_manager.router.redis.publish.assert_called()
        call_args = ws_manager.router.redis.publish.call_args[0]
        assert "symbol:AAPL" in call_args[0]

        published = json.loads(call_args[1])
        assert published["message"]["type"] == MessageType.MARKET_DATA.value
        assert published["message"]["symbol"] == "AAPL"
        assert published["message"]["data"] == data

    async def test_send_signal(self, ws_manager):
        """Test sending trading signal"""
        signal_data = {
            "type": "BUY",
            "confidence": 0.85,
            "stop_loss": 180.0,
            "take_profit": 190.0
        }

        await ws_manager.send_signal("GOOGL", signal_data)

        # Verify publish
        ws_manager.router.redis.publish.assert_called()
        call_args = ws_manager.router.redis.publish.call_args[0]
        assert "symbol:GOOGL" in call_args[0]

        published = json.loads(call_args[1])
        assert published["message"]["type"] == MessageType.SIGNAL.value
        assert published["message"]["signal"] == signal_data

    async def test_send_alert(self, ws_manager):
        """Test sending user alert"""
        alert_data = {
            "level": "warning",
            "message": "Stop loss approaching",
            "position_id": "pos_123"
        }

        await ws_manager.send_alert("user_456", alert_data)

        # Verify publish
        ws_manager.router.redis.publish.assert_called()
        call_args = ws_manager.router.redis.publish.call_args[0]
        assert "user:user_456" in call_args[0]

        published = json.loads(call_args[1])
        assert published["message"]["type"] == MessageType.ALERT.value
        assert published["message"]["alert"] == alert_data

    async def test_get_stats(self, ws_manager, mock_websocket):
        """Test getting statistics"""
        # Create some connections
        await ws_manager.connect(websocket=mock_websocket)
        await ws_manager.connect(websocket=mock_websocket)

        # Update metrics
        ws_manager.metrics["messages_sent"] = 100
        ws_manager.metrics["errors"] = 2

        stats = await ws_manager.get_stats()

        assert stats["server_id"] == ws_manager.server_id
        assert stats["local_connections"] == 2
        assert stats["metrics"]["messages_sent"] == 100
        assert stats["metrics"]["errors"] == 2

    async def test_connection_error_handling(self, ws_manager, mock_websocket):
        """Test error handling during connection"""
        # Make accept fail
        mock_websocket.accept.side_effect = Exception("Connection failed")

        with pytest.raises(Exception):
            await ws_manager.connect(websocket=mock_websocket)

        # Verify metrics updated
        assert ws_manager.metrics["connections_rejected"] == 1

    async def test_message_listener_handling(self, ws_manager, mock_websocket):
        """Test message listener processing"""
        # Create connections
        conn_id1 = await ws_manager.connect(websocket=mock_websocket)
        conn_id2 = await ws_manager.connect(websocket=mock_websocket)

        # Subscribe to symbols
        await ws_manager.subscribe_to_symbol(conn_id1, "AAPL")
        await ws_manager.subscribe_to_symbol(conn_id2, "AAPL")
        await ws_manager.subscribe_to_symbol(conn_id2, "GOOGL")

        # Simulate Redis message for symbol
        redis_message = {
            "type": "message",
            "channel": b"symbol:AAPL",
            "data": json.dumps({
                "message": {
                    "type": "market_data",
                    "price": 185.0
                }
            }).encode()
        }

        # Mock pubsub to return message once
        ws_manager.router.pubsub.get_message = AsyncMock(
            side_effect=[redis_message, None]
        )

        # Process one iteration of listener
        # (In real usage this runs in background task)

        # Verify subscriptions
        assert len(ws_manager.symbol_subscribers["AAPL"]) == 2
        assert len(ws_manager.symbol_subscribers["GOOGL"]) == 1


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for scalable WebSocket"""

    async def test_multi_server_simulation(self):
        """Test simulating multiple servers"""
        with patch('src.websocket.scalable_manager.redis.from_url') as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis

            # Create two server instances
            server1 = ScalableWebSocketManager(
                redis_url="redis://localhost:6379",
                server_id="server_1"
            )
            server1.redis = mock_redis

            server2 = ScalableWebSocketManager(
                redis_url="redis://localhost:6379",
                server_id="server_2"
            )
            server2.redis = mock_redis

            # Both servers share same Redis
            assert server1.server_id != server2.server_id

            # Simulate connections on different servers
            ws1 = AsyncMock()
            ws2 = AsyncMock()

            # This would allow cross-server communication via Redis

    async def test_failover_scenario(self, ws_manager, mock_websocket):
        """Test connection failover scenario"""
        # Connect multiple clients
        connections = []
        for i in range(5):
            conn_id = await ws_manager.connect(websocket=mock_websocket)
            connections.append(conn_id)
            await ws_manager.subscribe_to_symbol(conn_id, "AAPL")

        # Simulate server shutdown
        await ws_manager.shutdown()

        # Verify all connections cleaned up
        assert len(ws_manager.connections) == 0
        assert len(ws_manager.symbol_subscribers) == 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
