"""
Golden Eye API Routes - RESTful and WebSocket endpoints for the chat interface
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.middleware.auth import get_current_user
from src.middleware.rate_limiter import RateLimiter
from src.services.golden_eye_orchestrator import (
    GoldenEyeContext,
    QueryIntent,
    golden_eye_orchestrator,
)
from src.services.mcp_agent_tools import (
    analyze_with_agent,
    execute_agent_workflow,
    get_agent_consensus,
    predict_with_ai,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/golden-eye", tags=["golden-eye"])

# Rate limiter for Golden Eye endpoints
golden_eye_limiter = RateLimiter(max_requests=60, window_seconds=60)  # 60 requests per minute


# Request/Response models
class GoldenEyeQueryRequest(BaseModel):
    """Request model for Golden Eye queries"""

    query: str = Field(..., description="User's natural language query")
    symbol: str = Field(..., description="Trading symbol context")
    timeframe: str = Field(default="1h", description="Chart timeframe")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class GoldenEyeQueryResponse(BaseModel):
    """Response model for Golden Eye queries"""

    query_id: str
    status: str
    message: str


class AgentDiscoveryResponse(BaseModel):
    """Response model for agent discovery"""

    agents: Dict[str, Dict[str, Any]]
    total: int
    categories: Dict[str, List[str]]


class MCPToolDiscoveryResponse(BaseModel):
    """Response model for MCP tool discovery"""

    tools: List[Dict[str, Any]]
    total: int
    categories: Dict[str, List[str]]


# REST Endpoints


@router.post("/query")
async def process_query(
    request: GoldenEyeQueryRequest, user=Depends(get_current_user)
) -> GoldenEyeQueryResponse:
    """
    Process a Golden Eye query (non-streaming version)
    Returns a query ID that can be used to retrieve results
    """
    # Apply rate limiting
    if not await golden_eye_limiter.check_rate_limit(user.id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    query_id = str(uuid.uuid4())

    # Start processing in background
    asyncio.create_task(
        _process_query_background(
            query_id=query_id,
            query=request.query,
            context=GoldenEyeContext(
                symbol=request.symbol,
                timeframe=request.timeframe,
                user_id=user.id,
                preferences=request.context,
            ),
        )
    )

    return GoldenEyeQueryResponse(
        query_id=query_id,
        status="processing",
        message="Query is being processed. Use WebSocket or polling to get results.",
    )


@router.post("/query/stream")
async def process_query_stream(request: GoldenEyeQueryRequest, user=Depends(get_current_user)):
    """
    Process a Golden Eye query with Server-Sent Events streaming
    """
    # Apply rate limiting
    if not await golden_eye_limiter.check_rate_limit(user.id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    async def event_generator():
        """Generate SSE events"""
        context = GoldenEyeContext(
            symbol=request.symbol,
            timeframe=request.timeframe,
            user_id=user.id,
            preferences=request.context,
        )

        try:
            async for event in golden_eye_orchestrator.process_query(request.query, context):
                # Format as SSE
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming query: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield 'data: {"type": "complete"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


@router.get("/agents/discover")
async def discover_agents(
    category: Optional[str] = None, user=Depends(get_current_user)
) -> AgentDiscoveryResponse:
    """
    Discover available agents and their capabilities
    """
    from src.services.agent_registry import agent_registry
    from src.services.mcp_agent_tools import AGENT_TYPE_MAPPING

    all_agents = agent_registry.get_all_agents()

    # Build agent info
    agent_info = {}
    categories = {}

    for agent_name, agent_class in all_agents.items():
        agent_type = AGENT_TYPE_MAPPING.get(agent_name, "unknown")

        info = {
            "name": agent_name,
            "type": agent_type.value if hasattr(agent_type, "value") else str(agent_type),
            "description": agent_class.__doc__ or "Trading analysis agent",
            "mcp_tool": f"analyze_with_{agent_name.lower()}",
            "capabilities": getattr(agent_class, "capabilities", []),
        }

        agent_info[agent_name] = info

        # Categorize
        category_name = info["type"]
        if category_name not in categories:
            categories[category_name] = []
        categories[category_name].append(agent_name)

    # Filter by category if specified
    if category:
        agent_info = {name: info for name, info in agent_info.items() if info["type"] == category}

    return AgentDiscoveryResponse(agents=agent_info, total=len(agent_info), categories=categories)


@router.get("/tools/discover")
async def discover_mcp_tools(
    tool_type: Optional[str] = None, user=Depends(get_current_user)
) -> MCPToolDiscoveryResponse:
    """
    Discover available MCP tools
    """
    from src.services.mcp_tools import get_all_mcp_tools

    all_tools = get_all_mcp_tools()

    # Filter by type if specified
    if tool_type:
        all_tools = [t for t in all_tools if t.type.value == tool_type]

    # Categorize tools
    categories = {}
    for tool in all_tools:
        category = tool.type.value
        if category not in categories:
            categories[category] = []
        categories[category].append(tool.name)

    # Convert to dict format
    tool_dicts = [tool.to_dict() for tool in all_tools]

    return MCPToolDiscoveryResponse(tools=tool_dicts, total=len(tool_dicts), categories=categories)


@router.post("/agents/consensus")
async def get_consensus(
    symbol: str,
    agents: List[str],
    timeframe: str = "1h",
    voting_method: str = "weighted",
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get consensus from multiple agents
    """
    result = await get_agent_consensus(
        symbol=symbol, agents=agents, timeframe=timeframe, voting_method=voting_method
    )

    return result


@router.post("/predict")
async def generate_prediction(
    symbol: str, horizon: int = 24, use_ensemble: bool = True, user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate AI-enhanced prediction
    """
    result = await predict_with_ai(symbol=symbol, horizon=horizon, use_ensemble=use_ensemble)

    return result


@router.post("/workflow/{workflow_name}")
async def execute_workflow(
    workflow_name: str,
    symbol: str,
    parameters: Optional[Dict[str, Any]] = None,
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Execute a predefined workflow
    """
    result = await execute_agent_workflow(
        workflow_name=workflow_name, symbol=symbol, parameters=parameters
    )

    return result


# WebSocket endpoint for real-time chat


class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)

    async def broadcast(self, message: dict, exclude: Optional[str] = None):
        for client_id, websocket in self.active_connections.items():
            if client_id != exclude:
                await websocket.send_json(message)


manager = ConnectionManager()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time Golden Eye chat
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            # Validate message
            if "type" not in data:
                await websocket.send_json(
                    {"type": "error", "message": "Message must include 'type' field"}
                )
                continue

            # Handle different message types
            if data["type"] == "query":
                await _handle_websocket_query(websocket, client_id, data)

            elif data["type"] == "ping":
                await websocket.send_json({"type": "pong"})

            elif data["type"] == "agent_status":
                await _handle_agent_status_request(websocket, data)

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {data['type']}"}
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        manager.disconnect(client_id)


async def _handle_websocket_query(websocket: WebSocket, client_id: str, data: dict):
    """Handle query message over WebSocket"""
    try:
        # Extract query parameters
        query = data.get("query", "")
        symbol = data.get("symbol", "AAPL")
        timeframe = data.get("timeframe", "1h")
        context_data = data.get("context", {})

        # Create context
        context = GoldenEyeContext(
            symbol=symbol, timeframe=timeframe, user_id=client_id, preferences=context_data
        )

        # Process query and stream results
        async for event in golden_eye_orchestrator.process_query(query, context):
            await websocket.send_json(
                {"type": "query_event", "event": event, "timestamp": datetime.now().isoformat()}
            )

        # Send completion message
        await websocket.send_json(
            {"type": "query_complete", "timestamp": datetime.now().isoformat()}
        )

    except Exception as e:
        logger.error(f"Error handling WebSocket query: {str(e)}")
        await websocket.send_json({"type": "error", "message": str(e)})


async def _handle_agent_status_request(websocket: WebSocket, data: dict):
    """Handle agent status request"""
    from src.services.agent_registry import agent_registry

    agents = data.get("agents", [])
    if not agents:
        agents = list(agent_registry.get_all_agents().keys())

    status_info = {}
    for agent_name in agents:
        if agent_registry.has_agent(agent_name):
            status_info[agent_name] = {
                "available": True,
                "type": AGENT_TYPE_MAPPING.get(agent_name, "unknown").value,
                "last_update": datetime.now().isoformat(),
            }
        else:
            status_info[agent_name] = {"available": False, "error": "Agent not found"}

    await websocket.send_json(
        {
            "type": "agent_status_response",
            "agents": status_info,
            "timestamp": datetime.now().isoformat(),
        }
    )


async def _process_query_background(query_id: str, query: str, context: GoldenEyeContext):
    """Process query in background (for non-streaming endpoint)"""
    try:
        results = []
        async for event in golden_eye_orchestrator.process_query(query, context):
            results.append(event)

        # Store results in cache for retrieval
        from src.services.redis_cache_service import redis_cache

        await redis_cache.set(
            f"golden_eye_query:{query_id}",
            {
                "query": query,
                "context": context.__dict__,
                "results": results,
                "status": "complete",
                "timestamp": datetime.now().isoformat(),
            },
            ttl=3600,  # Keep for 1 hour
        )

    except Exception as e:
        logger.error(f"Error processing query {query_id}: {str(e)}")
        await redis_cache.set(
            f"golden_eye_query:{query_id}",
            {
                "query": query,
                "error": str(e),
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            },
            ttl=3600,
        )


@router.get("/query/{query_id}/status")
async def get_query_status(query_id: str, user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get status of a query by ID
    """
    from src.services.redis_cache_service import redis_cache

    result = await redis_cache.get(f"golden_eye_query:{query_id}")

    if not result:
        raise HTTPException(status_code=404, detail="Query not found")

    return result


# Health check endpoint
@router.get("/health")
async def health_check():
    """Check Golden Eye service health"""
    return {
        "status": "healthy",
        "service": "golden-eye",
        "timestamp": datetime.now().isoformat(),
        "features": {"streaming": True, "websocket": True, "mcp_tools": True, "multi_llm": True},
    }


__all__ = ["router"]
