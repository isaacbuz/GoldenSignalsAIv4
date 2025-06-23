# MCP Implementation Guide for GoldenSignalsAI

## Quick Start Implementation

This guide provides practical steps to implement MCP in your GoldenSignalsAI project, starting with the most valuable components.

## Phase 1: Core MCP Server (Week 1)

### Step 1: Install MCP SDK

```bash
# Install MCP Python SDK
pip install mcp

# Install additional dependencies
pip install fastapi uvicorn redis asyncpg
```

### Step 2: Create First MCP Server - Trading Signals

```python
# mcp_servers/trading_signals_server.py
import asyncio
from typing import Any
from mcp.server import Server
from mcp.server.models import InitializeResult
from mcp.types import Tool, TextContent, Resource
import json

# Import your existing orchestrator
from agents.orchestration.simple_orchestrator import SimpleOrchestrator

class TradingSignalsMCP(Server):
    """MCP server that exposes your trading signals"""
    
    def __init__(self):
        super().__init__("goldensignals-trading")
        self.orchestrator = SimpleOrchestrator()
        
    async def handle_initialize(self) -> InitializeResult:
        """Initialize the MCP server with capabilities"""
        return InitializeResult(
            protocol_version="2024-11-05",
            capabilities={
                "tools": {"list_changed": False},
                "resources": {"subscribe": True, "list_changed": True}
            },
            server_info={
                "name": "GoldenSignals Trading Server",
                "version": "1.0.0"
            }
        )
    
    async def handle_list_tools(self) -> list[Tool]:
        """List available trading tools"""
        return [
            Tool(
                name="generate_signal",
                description="Generate trading signal for a symbol using all agents",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_agent_breakdown",
                description="Get detailed breakdown of all agent signals",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"}
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="analyze_pattern",
                description="Analyze chart patterns for a symbol",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "timeframe": {
                            "type": "string",
                            "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]
                        }
                    },
                    "required": ["symbol", "timeframe"]
                }
            )
        ]
    
    async def handle_call_tool(self, name: str, arguments: dict) -> list[TextContent]:
        """Execute tool calls"""
        
        if name == "generate_signal":
            signal = self.orchestrator.generate_signals_for_symbol(
                arguments["symbol"]
            )
            result = self.orchestrator.to_json(signal)
            
        elif name == "get_agent_breakdown":
            signal = self.orchestrator.generate_signals_for_symbol(
                arguments["symbol"]
            )
            result = {
                "symbol": arguments["symbol"],
                "consensus": {
                    "action": signal["action"],
                    "confidence": signal["confidence"]
                },
                "agents": signal["metadata"]["agent_breakdown"]
            }
            
        elif name == "analyze_pattern":
            # Use your pattern recognition agents
            from agents.core.technical.pattern_recognition_agent import PatternAgent
            pattern_agent = PatternAgent()
            result = pattern_agent.analyze(
                arguments["symbol"],
                arguments["timeframe"]
            )
            
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    async def handle_list_resources(self) -> list[Resource]:
        """List available resources for subscription"""
        return [
            Resource(
                uri="signals://realtime",
                name="Real-time Trading Signals",
                description="Subscribe to real-time trading signals",
                mime_type="application/json"
            ),
            Resource(
                uri="signals://agent-insights",
                name="Agent Insights Stream",
                description="Real-time insights from all trading agents",
                mime_type="application/json"
            )
        ]

# Run the server
if __name__ == "__main__":
    import mcp.server.stdio
    
    async def main():
        server = TradingSignalsMCP()
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.handle_initialize,
                server.handle_list_tools,
                server.handle_call_tool,
                server.handle_list_resources
            )
    
    asyncio.run(main())
```

### Step 3: Create MCP Configuration

```json
// mcp_config.json
{
  "mcpServers": {
    "goldensignals": {
      "command": "python",
      "args": ["mcp_servers/trading_signals_server.py"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

### Step 4: Test with Claude Desktop

1. Copy the config to Claude's configuration directory:
```bash
# macOS
cp mcp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
cp mcp_config.json %APPDATA%\Claude\claude_desktop_config.json
```

2. Restart Claude Desktop
3. Test by asking: "Generate a trading signal for AAPL"

## Phase 2: Market Data MCP Server (Week 2)

### Step 1: Create Market Data Server

```python
# mcp_servers/market_data_server.py
from mcp.server import Server
from mcp.types import Resource
import yfinance as yf
import json

class MarketDataMCP(Server):
    """MCP server for market data access"""
    
    def __init__(self):
        super().__init__("goldensignals-market-data")
        self.cache = {}
        
    async def handle_list_resources(self) -> list[Resource]:
        return [
            Resource(
                uri="market://quotes/AAPL",
                name="AAPL Real-time Quote",
                description="Real-time quote for Apple Inc.",
                mime_type="application/json"
            ),
            Resource(
                uri="market://historical/AAPL/1d",
                name="AAPL Daily History",
                description="Historical daily data for Apple Inc.",
                mime_type="application/json"
            )
        ]
    
    async def handle_read_resource(self, uri: str) -> str:
        """Read market data resources"""
        parts = uri.split("/")
        
        if parts[2] == "quotes":
            symbol = parts[3]
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return json.dumps({
                "symbol": symbol,
                "price": info.get("currentPrice", 0),
                "change": info.get("regularMarketChange", 0),
                "changePercent": info.get("regularMarketChangePercent", 0),
                "volume": info.get("volume", 0),
                "timestamp": datetime.now().isoformat()
            })
            
        elif parts[2] == "historical":
            symbol = parts[3]
            period = parts[4] if len(parts) > 4 else "1d"
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            return json.dumps({
                "symbol": symbol,
                "period": period,
                "data": hist.to_dict(orient="records"),
                "timestamp": datetime.now().isoformat()
            })
```

## Phase 3: Secure Gateway (Week 3)

### Step 1: Create MCP Gateway with Authentication

```python
# gateway/mcp_gateway.py
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from typing import Optional
import httpx

app = FastAPI(title="GoldenSignals MCP Gateway")
security = HTTPBearer()

# MCP server registry
MCP_SERVERS = {
    "trading": "http://localhost:8001",
    "market-data": "http://localhost:8002",
    "sentiment": "http://localhost:8003"
}

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/mcp/{server}/{method}")
async def mcp_proxy(
    server: str,
    method: str,
    request_body: dict,
    user = Depends(verify_token)
):
    """Proxy MCP requests with authentication"""
    
    if server not in MCP_SERVERS:
        raise HTTPException(status_code=404, detail=f"Server {server} not found")
    
    # Check permissions
    required_permission = f"{server}:{method}"
    if required_permission not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Forward request to MCP server
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MCP_SERVERS[server]}/{method}",
            json=request_body,
            headers={"X-User-ID": user["sub"]}
        )
        
    return response.json()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "servers": list(MCP_SERVERS.keys())}
```

### Step 2: Add Rate Limiting

```python
# gateway/rate_limiter.py
from fastapi import HTTPException
import redis
import time

class RateLimiter:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        
    async def check_rate_limit(self, user_id: str, limit: int = 100, window: int = 60):
        """Check if user has exceeded rate limit"""
        key = f"rate_limit:{user_id}"
        current_time = int(time.time())
        window_start = current_time - window
        
        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count requests in window
        request_count = self.redis.zcard(key)
        
        if request_count >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {limit} requests per {window} seconds"
            )
        
        # Add current request
        self.redis.zadd(key, {str(current_time): current_time})
        self.redis.expire(key, window)
        
        return True
```

## Phase 4: Agent Integration (Week 4)

### Step 1: Wrap Existing Agents

```python
# mcp_servers/agent_wrapper.py
from typing import Dict, Any
from mcp.types import Tool

class AgentMCPWrapper:
    """Generic wrapper to expose any agent as MCP tool"""
    
    def __init__(self, agent, agent_name: str):
        self.agent = agent
        self.agent_name = agent_name
        
    def to_mcp_tool(self) -> Tool:
        """Convert agent to MCP tool"""
        return Tool(
            name=f"{self.agent_name}_signal",
            description=f"Generate signal using {self.agent_name}",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "params": {
                        "type": "object",
                        "description": "Agent-specific parameters"
                    }
                },
                "required": ["symbol"]
            }
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent and return result"""
        symbol = arguments["symbol"]
        params = arguments.get("params", {})
        
        # Call your existing agent
        signal = self.agent.generate_signal(symbol, **params)
        
        return {
            "agent": self.agent_name,
            "symbol": symbol,
            "signal": signal,
            "timestamp": datetime.now().isoformat()
        }

# Example: Wrap all your agents
def create_agent_tools():
    """Create MCP tools for all agents"""
    from agents.core.technical.simple_working_agent import SimpleRSIAgent
    from agents.core.technical.macd_agent import MACDAgent
    # ... import other agents
    
    tools = []
    
    # Wrap each agent
    rsi_wrapper = AgentMCPWrapper(SimpleRSIAgent(), "rsi")
    tools.append(rsi_wrapper.to_mcp_tool())
    
    macd_wrapper = AgentMCPWrapper(MACDAgent(), "macd")
    tools.append(macd_wrapper.to_mcp_tool())
    
    # ... wrap other agents
    
    return tools
```

## Phase 5: Production Deployment

### Step 1: Docker Compose Setup

```yaml
# docker-compose.mcp.yml
version: '3.8'

services:
  mcp-gateway:
    build: ./gateway
    ports:
      - "8000:8000"
    environment:
      - JWT_SECRET=${JWT_SECRET}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - mcp-trading
      - mcp-market-data

  mcp-trading:
    build: ./mcp_servers/trading
    environment:
      - PYTHONPATH=/app
      - DB_URL=postgresql://postgres:5432/goldensignals
    depends_on:
      - postgres

  mcp-market-data:
    build: ./mcp_servers/market_data
    environment:
      - CACHE_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=goldensignals
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

### Step 2: Monitoring Setup

```python
# monitoring/mcp_metrics.py
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# Define metrics
mcp_requests = Counter(
    'mcp_requests_total',
    'Total MCP requests',
    ['server', 'method', 'status']
)

mcp_latency = Histogram(
    'mcp_request_duration_seconds',
    'MCP request latency',
    ['server', 'method']
)

# Add to your gateway
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

# Use in request handler
@app.post("/mcp/{server}/{method}")
async def mcp_proxy_with_metrics(server: str, method: str, ...):
    start_time = time.time()
    
    try:
        result = await mcp_proxy(server, method, ...)
        mcp_requests.labels(server=server, method=method, status="success").inc()
        return result
    except Exception as e:
        mcp_requests.labels(server=server, method=method, status="error").inc()
        raise
    finally:
        mcp_latency.labels(server=server, method=method).observe(
            time.time() - start_time
        )
```

## Testing Your MCP Implementation

### 1. Unit Test Example

```python
# tests/test_mcp_trading.py
import pytest
from mcp_servers.trading_signals_server import TradingSignalsMCP

@pytest.mark.asyncio
async def test_generate_signal():
    server = TradingSignalsMCP()
    
    # Test tool listing
    tools = await server.handle_list_tools()
    assert len(tools) == 3
    assert any(t.name == "generate_signal" for t in tools)
    
    # Test signal generation
    result = await server.handle_call_tool(
        "generate_signal",
        {"symbol": "AAPL"}
    )
    
    assert len(result) == 1
    assert result[0].type == "text"
    
    # Parse result
    import json
    signal_data = json.loads(result[0].text)
    assert "signal_type" in signal_data
    assert "confidence" in signal_data
```

### 2. Integration Test

```python
# tests/test_mcp_integration.py
import httpx
import pytest

@pytest.mark.asyncio
async def test_end_to_end():
    """Test complete MCP flow through gateway"""
    
    # Get auth token
    token = "your-test-jwt-token"
    
    async with httpx.AsyncClient() as client:
        # Call through gateway
        response = await client.post(
            "http://localhost:8000/mcp/trading/generate_signal",
            json={"symbol": "AAPL"},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["signal_type"] in ["BUY", "SELL", "HOLD"]
```

## Best Practices

1. **Start Small**: Begin with one MCP server (trading signals)
2. **Use Existing Code**: Wrap your current agents, don't rewrite
3. **Add Security Gradually**: Start with basic auth, enhance over time
4. **Monitor Everything**: Add metrics from day one
5. **Test Thoroughly**: Each MCP server should have comprehensive tests

## Common Issues and Solutions

### Issue 1: Claude Desktop Can't Find Server
```bash
# Check logs
tail -f ~/Library/Logs/Claude/mcp.log

# Verify Python path
export PYTHONPATH=/path/to/your/project
```

### Issue 2: Performance Issues
```python
# Add caching
from functools import lru_cache

@lru_cache(maxsize=100)
async def get_cached_signal(symbol: str) -> dict:
    return await generate_signal(symbol)
```

### Issue 3: Authentication Errors
```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Log token validation
logger.debug(f"Token payload: {payload}")
```

## Next Steps

1. **Week 1**: Implement basic trading signals MCP server
2. **Week 2**: Add market data server and test with Claude
3. **Week 3**: Implement secure gateway with auth
4. **Week 4**: Wrap all agents and add monitoring
5. **Week 5**: Production deployment and optimization

This implementation guide provides a practical path to adding MCP to your GoldenSignalsAI project without disrupting your existing architecture. 