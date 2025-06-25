"""
MCP Client Library
For connecting to MCP servers
"""

import asyncio
import websockets
import json
from typing import Dict, Any, Optional

class MCPClient:
    """Client for connecting to MCP servers"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.websocket = None
        
    async def connect(self):
        """Connect to MCP server"""
        self.websocket = await websockets.connect(self.server_url)
        
    async def disconnect(self):
        """Disconnect from server"""
        if self.websocket:
            await self.websocket.close()
            
    async def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send request to server"""
        if not self.websocket:
            raise Exception("Not connected to server")
            
        request = {
            "method": method,
            "params": params or {}
        }
        
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        
        return json.loads(response)
    
    async def get_capabilities(self) -> List[str]:
        """Get server capabilities"""
        response = await self.request("capabilities")
        return response.get("capabilities", [])

# Example usage
async def example():
    # Connect to RAG Query server
    rag_client = MCPClient("ws://localhost:8501")
    await rag_client.connect()
    
    # Get capabilities
    capabilities = await rag_client.get_capabilities()
    print(f"RAG Server capabilities: {capabilities}")
    
    # Query RAG
    result = await rag_client.request("rag.query", {
        "query": "What are the historical patterns for AAPL?",
        "k": 3
    })
    print(f"RAG Query result: {result}")
    
    await rag_client.disconnect()

if __name__ == "__main__":
    asyncio.run(example())
