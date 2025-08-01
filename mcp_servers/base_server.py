"""
MCP (Model Context Protocol) Base Server
Provides foundation for specialized MCP servers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio
import json
import logging
from datetime import datetime
import websockets

logger = logging.getLogger(__name__)

class MCPServer(ABC):
    """Base class for MCP servers"""

    def __init__(self, name: str, port: int):
        self.name = name
        self.port = port
        self.clients = set()
        self.capabilities = []

    @abstractmethod
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request"""
        pass

    async def start(self):
        """Start MCP server"""
        logger.info(f"Starting {self.name} MCP server on port {self.port}")
        await websockets.serve(self.handle_client, "localhost", self.port)

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                request = json.loads(message)
                response = await self.handle_request(request)
                await websocket.send(json.dumps(response))
        finally:
            self.clients.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all clients"""
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients]
            )
