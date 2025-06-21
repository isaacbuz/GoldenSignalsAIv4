#!/usr/bin/env python3
"""
🚀 GoldenSignalsAI - WebSocket Service
Real-time market data streaming with automatic fallback mechanisms

Features:
- WebSocket-first architecture
- Automatic fallback to SSE/polling
- Connection management
- Heartbeat/keepalive
- Automatic reconnection
- Message queuing
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
from websockets.exceptions import WebSocketException
import aiohttp
from asyncio import Queue
import uuid

from src.services.rate_limit_handler import get_rate_limit_handler, RequestPriority

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

class MessageType(Enum):
    """WebSocket message types"""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    UPDATE = "update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SNAPSHOT = "snapshot"

@dataclass
class MarketUpdate:
    """Market data update"""
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    timestamp: str
    change: float
    change_percent: float

@dataclass
class Subscription:
    """Subscription details"""
    id: str
    symbols: List[str]
    channels: List[str]
    callback: Optional[Callable]
    created_at: float

class WebSocketService:
    """WebSocket service with automatic fallback"""
    
    def __init__(self, ws_url: str = "wss://stream.goldensignals.ai/v1/market"):
        self.ws_url = ws_url
        self.ws = None
        self.state = ConnectionState.DISCONNECTED
        self.subscriptions: Dict[str, Subscription] = {}
        self.symbol_callbacks: Dict[str, Set[Callable]] = {}
        self.message_queue: Queue = Queue()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1.0  # Start with 1 second
        self.heartbeat_interval = 30
        self.last_heartbeat = time.time()
        self.rate_limit_handler = get_rate_limit_handler()
        
        # Fallback options
        self.use_sse = True
        self.use_polling = True
        self.polling_interval = 1.0  # 1 second
        
        # Background tasks will be started when needed
        self._background_tasks_started = False
    
    def _ensure_background_tasks(self):
        """Ensure background tasks are started when in async context"""
        if not self._background_tasks_started:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._connection_manager())
                loop.create_task(self._heartbeat_task())
                loop.create_task(self._message_processor())
                self._background_tasks_started = True
            except RuntimeError:
                # No event loop running yet, tasks will start when one is available
                pass
    
    async def connect(self) -> bool:
        """Connect to WebSocket server"""
        # Ensure background tasks are running
        self._ensure_background_tasks()
        
        if self.state == ConnectionState.CONNECTED:
            return True
        
        self.state = ConnectionState.CONNECTING
        
        try:
            logger.info(f"Connecting to WebSocket: {self.ws_url}")
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.state = ConnectionState.CONNECTED
            self.reconnect_attempts = 0
            self.reconnect_delay = 1.0
            
            logger.info("✅ WebSocket connected successfully")
            
            # Resubscribe to all active subscriptions
            await self._resubscribe_all()
            
            # Start receiving messages
            asyncio.create_task(self._receive_messages())
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.state = ConnectionState.FAILED
            await self._fallback_to_alternative()
            return False
    
    async def _receive_messages(self):
        """Receive messages from WebSocket"""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            self.state = ConnectionState.DISCONNECTED
            await self._reconnect()
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        msg_type = data.get("type")
        
        if msg_type == MessageType.UPDATE.value:
            await self._handle_market_update(data)
        elif msg_type == MessageType.HEARTBEAT.value:
            self.last_heartbeat = time.time()
        elif msg_type == MessageType.ERROR.value:
            logger.error(f"Server error: {data.get('message')}")
        elif msg_type == MessageType.SNAPSHOT.value:
            await self._handle_snapshot(data)
    
    async def _handle_market_update(self, data: Dict[str, Any]):
        """Handle market data update"""
        symbol = data.get("symbol")
        if not symbol:
            return
        
        update = MarketUpdate(
            symbol=symbol,
            price=data.get("price", 0),
            volume=data.get("volume", 0),
            bid=data.get("bid", 0),
            ask=data.get("ask", 0),
            timestamp=data.get("timestamp"),
            change=data.get("change", 0),
            change_percent=data.get("change_percent", 0)
        )
        
        # Call all callbacks for this symbol
        if symbol in self.symbol_callbacks:
            for callback in self.symbol_callbacks[symbol]:
                try:
                    await callback(update)
                except Exception as e:
                    logger.error(f"Callback error for {symbol}: {e}")
    
    async def subscribe(
        self,
        symbols: List[str],
        channels: List[str] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """Subscribe to market data"""
        # Ensure background tasks are running
        self._ensure_background_tasks()
        
        if channels is None:
            channels = ["trade", "quote", "bar"]
        
        sub_id = str(uuid.uuid4())
        subscription = Subscription(
            id=sub_id,
            symbols=symbols,
            channels=channels,
            callback=callback,
            created_at=time.time()
        )
        
        self.subscriptions[sub_id] = subscription
        
        # Register callbacks
        if callback:
            for symbol in symbols:
                if symbol not in self.symbol_callbacks:
                    self.symbol_callbacks[symbol] = set()
                self.symbol_callbacks[symbol].add(callback)
        
        # Send subscription message if connected
        if self.state == ConnectionState.CONNECTED:
            await self._send_subscription(subscription)
        
        return sub_id
    
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from market data"""
        if subscription_id not in self.subscriptions:
            return
        
        subscription = self.subscriptions[subscription_id]
        
        # Remove callbacks
        if subscription.callback:
            for symbol in subscription.symbols:
                if symbol in self.symbol_callbacks:
                    self.symbol_callbacks[symbol].discard(subscription.callback)
                    if not self.symbol_callbacks[symbol]:
                        del self.symbol_callbacks[symbol]
        
        # Send unsubscribe message if connected
        if self.state == ConnectionState.CONNECTED:
            await self._send_unsubscription(subscription)
        
        del self.subscriptions[subscription_id]
    
    async def _send_subscription(self, subscription: Subscription):
        """Send subscription message to server"""
        if not self.ws:
            return
        
        message = {
            "type": MessageType.SUBSCRIBE.value,
            "id": subscription.id,
            "symbols": subscription.symbols,
            "channels": subscription.channels
        }
        
        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send subscription: {e}")
    
    async def _send_unsubscription(self, subscription: Subscription):
        """Send unsubscribe message to server"""
        if not self.ws:
            return
        
        message = {
            "type": MessageType.UNSUBSCRIBE.value,
            "id": subscription.id,
            "symbols": subscription.symbols
        }
        
        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send unsubscription: {e}")
    
    async def _resubscribe_all(self):
        """Resubscribe to all active subscriptions"""
        for subscription in self.subscriptions.values():
            await self._send_subscription(subscription)
    
    async def _connection_manager(self):
        """Manage WebSocket connection lifecycle"""
        while True:
            try:
                if self.state == ConnectionState.DISCONNECTED:
                    await self.connect()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Connection manager error: {e}")
                await asyncio.sleep(5)
    
    async def _heartbeat_task(self):
        """Send periodic heartbeats"""
        while True:
            try:
                if self.state == ConnectionState.CONNECTED and self.ws:
                    # Check if we need to send heartbeat
                    if time.time() - self.last_heartbeat > self.heartbeat_interval:
                        message = {
                            "type": MessageType.HEARTBEAT.value,
                            "timestamp": time.time()
                        }
                        await self.ws.send(json.dumps(message))
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _reconnect(self):
        """Reconnect to WebSocket with exponential backoff"""
        if self.state == ConnectionState.RECONNECTING:
            return
        
        self.state = ConnectionState.RECONNECTING
        
        while self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 60)
            
            logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})")
            await asyncio.sleep(delay)
            
            if await self.connect():
                return
        
        logger.error("Max reconnection attempts reached, falling back to alternatives")
        await self._fallback_to_alternative()
    
    async def _fallback_to_alternative(self):
        """Fallback to SSE or polling"""
        if self.use_sse:
            logger.info("Falling back to Server-Sent Events")
            asyncio.create_task(self._sse_connection())
        elif self.use_polling:
            logger.info("Falling back to polling")
            asyncio.create_task(self._polling_loop())
    
    async def _sse_connection(self):
        """Server-Sent Events fallback"""
        sse_url = self.ws_url.replace("wss://", "https://").replace("/ws", "/sse")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(sse_url) as response:
                    async for line in response.content:
                        if line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                await self._handle_message(data)
                            except Exception as e:
                                logger.error(f"SSE parse error: {e}")
                                
        except Exception as e:
            logger.error(f"SSE connection failed: {e}")
            if self.use_polling:
                await self._polling_loop()
    
    async def _polling_loop(self):
        """Polling fallback"""
        logger.info("Starting polling fallback")
        
        while self.state != ConnectionState.CONNECTED:
            try:
                # Get all subscribed symbols
                all_symbols = set()
                for sub in self.subscriptions.values():
                    all_symbols.update(sub.symbols)
                
                if all_symbols:
                    # Fetch quotes using rate limit handler
                    quotes = await self.rate_limit_handler.batch_get_quotes(
                        list(all_symbols),
                        priority=RequestPriority.HIGH
                    )
                    
                    # Convert to market updates
                    for symbol, quote in quotes.items():
                        if quote:
                            update = MarketUpdate(
                                symbol=symbol,
                                price=quote.get("price", 0),
                                volume=quote.get("volume", 0),
                                bid=quote.get("bid", 0),
                                ask=quote.get("ask", 0),
                                timestamp=quote.get("timestamp"),
                                change=quote.get("change", 0),
                                change_percent=quote.get("change_percent", 0)
                            )
                            
                            # Call callbacks
                            if symbol in self.symbol_callbacks:
                                for callback in self.symbol_callbacks[symbol]:
                                    try:
                                        await callback(update)
                                    except Exception as e:
                                        logger.error(f"Polling callback error: {e}")
                
                await asyncio.sleep(self.polling_interval)
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(self.polling_interval * 2)
    
    async def _message_processor(self):
        """Process queued messages"""
        while True:
            try:
                message = await self.message_queue.get()
                # Process message
                await self._handle_message(message)
                
            except Exception as e:
                logger.error(f"Message processor error: {e}")
                await asyncio.sleep(1)
    
    async def close(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
        self.state = ConnectionState.DISCONNECTED

# Singleton instance
_websocket_service: Optional[WebSocketService] = None

def get_websocket_service() -> WebSocketService:
    """Get or create WebSocket service singleton"""
    global _websocket_service
    if _websocket_service is None:
        ws_url = "wss://stream.goldensignals.ai/v1/market"
        _websocket_service = WebSocketService(ws_url)
    return _websocket_service 