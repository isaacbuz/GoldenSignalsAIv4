"""
MCP Gateway Server - Central hub for all MCP requests
Provides authentication, rate limiting, load balancing, and audit logging
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import jwt
import httpx
import redis.asyncio as redis
from pydantic import BaseModel
import uvicorn
import os
from contextlib import asynccontextmanager

# Import timezone utilities
from src.utils.timezone_utils import now_utc

logger = logging.getLogger(__name__)

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = timedelta(hours=24)

# MCP Server Registry
MCP_SERVERS = {
    "trading-signals": {
        "url": "http://localhost:8001",
        "description": "Trading signals and agent coordination",
        "health_check": "/health"
    },
    "market-data": {
        "url": "http://localhost:8002",
        "description": "Real-time and historical market data",
        "health_check": "/health"
    },
    "portfolio": {
        "url": "http://localhost:8003",
        "description": "Portfolio management and risk analysis",
        "health_check": "/health"
    },
    "agent-bridge": {
        "url": "http://localhost:8004",
        "description": "Direct access to individual trading agents",
        "health_check": "/health"
    },
    "sentiment": {
        "url": "http://localhost:8005",
        "description": "Market sentiment and news analysis",
        "health_check": "/health"
    }
}

# Request/Response Models
class MCPRequest(BaseModel):
    method: str
    params: Dict[str, Any] = {}

class MCPResponse(BaseModel):
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class TokenData(BaseModel):
    username: str
    scopes: List[str] = []

# Security
security = HTTPBearer()

class RateLimiter:
    """Rate limiting implementation using Redis"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def check_rate_limit(
        self,
        user_id: str,
        resource: str,
        limit: int = 100,
        window: int = 60
    ) -> bool:
        """Check if user has exceeded rate limit"""
        key = f"rate_limit:{user_id}:{resource}"
        current_time = int(now_utc().timestamp())
        window_start = current_time - window

        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)

        # Count requests in window
        request_count = await self.redis.zcard(key)

        if request_count >= limit:
            return False

        # Add current request
        await self.redis.zadd(key, {str(current_time): current_time})
        await self.redis.expire(key, window)

        return True

class AuditLogger:
    """Audit logging for compliance and debugging"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def log_request(
        self,
        user: str,
        server: str,
        method: str,
        params: Dict[str, Any],
        ip_address: str
    ):
        """Log incoming request"""
        audit_entry = {
            "timestamp": now_utc().isoformat(),
            "user": user,
            "server": server,
            "method": method,
            "params": self._sanitize_params(params),
            "ip_address": ip_address,
            "type": "request"
        }

        # Store in Redis with TTL
        key = f"audit:{now_utc().strftime('%Y%m%d')}:{user}"
        await self.redis.lpush(key, json.dumps(audit_entry))
        await self.redis.expire(key, 86400 * 30)  # 30 days

    async def log_response(
        self,
        user: str,
        server: str,
        method: str,
        success: bool,
        latency_ms: float
    ):
        """Log response details"""
        audit_entry = {
            "timestamp": now_utc().isoformat(),
            "user": user,
            "server": server,
            "method": method,
            "success": success,
            "latency_ms": latency_ms,
            "type": "response"
        }

        key = f"audit:{now_utc().strftime('%Y%m%d')}:{user}"
        await self.redis.lpush(key, json.dumps(audit_entry))

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from parameters"""
        sanitized = params.copy()
        sensitive_fields = ["password", "api_key", "secret", "token", "credential"]

        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "[REDACTED]"

        return sanitized

class LoadBalancer:
    """Simple round-robin load balancer for MCP servers"""

    def __init__(self):
        self.server_indices = {}

    def get_next_server(self, server_type: str, servers: List[str]) -> str:
        """Get next server using round-robin"""
        if server_type not in self.server_indices:
            self.server_indices[server_type] = 0

        index = self.server_indices[server_type]
        server = servers[index % len(servers)]
        self.server_indices[server_type] = (index + 1) % len(servers)

        return server

# Global instances
redis_client: Optional[redis.Redis] = None
rate_limiter: Optional[RateLimiter] = None
audit_logger: Optional[AuditLogger] = None
load_balancer = LoadBalancer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global redis_client, rate_limiter, audit_logger

    # Startup
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = await redis.from_url(redis_url, decode_responses=True)
    rate_limiter = RateLimiter(redis_client)
    audit_logger = AuditLogger(redis_client)

    # Health check all MCP servers
    async with httpx.AsyncClient() as client:
        for server_name, server_info in MCP_SERVERS.items():
            try:
                response = await client.get(f"{server_info['url']}{server_info['health_check']}")
                if response.status_code == 200:
                    logger.info(f"✅ {server_name} server is healthy")
                else:
                    logger.warning(f"⚠️  {server_name} server returned {response.status_code}")
            except Exception as e:
                logger.error(f"❌ {server_name} server is not reachable: {e}")

    yield

    # Shutdown
    if redis_client:
        await redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="GoldenSignals MCP Gateway",
    description="Central gateway for Model Context Protocol servers",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication functions
def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = now_utc() + JWT_EXPIRATION
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Verify JWT token and extract user data"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        scopes = payload.get("scopes", [])
        return TokenData(username=username, scopes=scopes)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def check_permission(user: TokenData, server: str, method: str) -> bool:
    """Check if user has permission for server:method"""
    # Admin has all permissions
    if "admin" in user.scopes:
        return True

    # Check specific permission
    required_permission = f"{server}:{method}"
    if required_permission in user.scopes:
        return True

    # Check wildcard permission
    if f"{server}:*" in user.scopes:
        return True

    return False

# API Endpoints
@app.get("/")
async def root():
    """Gateway information"""
    return {
        "service": "GoldenSignals MCP Gateway",
        "version": "1.0.0",
        "servers": list(MCP_SERVERS.keys()),
        "timestamp": now_utc().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": now_utc().isoformat(),
        "servers": {}
    }

    # Check each MCP server
    async with httpx.AsyncClient() as client:
        for server_name, server_info in MCP_SERVERS.items():
            try:
                response = await client.get(
                    f"{server_info['url']}{server_info['health_check']}",
                    timeout=5.0
                )
                health_status["servers"][server_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            except Exception as e:
                health_status["servers"][server_name] = {
                    "status": "unreachable",
                    "error": str(e)
                }

    # Check if any server is unhealthy
    if any(s["status"] != "healthy" for s in health_status["servers"].values()):
        health_status["status"] = "degraded"

    return health_status

@app.post("/auth/login")
async def login(username: str, password: str):
    """Login endpoint - returns JWT token"""
    # TODO: Implement proper authentication
    # For now, simple demo authentication
    if username == "demo" and password == "demo123":
        scopes = ["trading-signals:*", "market-data:*", "portfolio:read"]
    elif username == "admin" and password == "admin123":
        scopes = ["admin"]
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(
        data={"sub": username, "scopes": scopes}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION.total_seconds()
    }

@app.post("/mcp/{server}/{method}")
async def mcp_proxy(
    server: str,
    method: str,
    request: MCPRequest,
    req: Request,
    user: TokenData = Depends(verify_token)
):
    """Proxy MCP requests to appropriate servers"""
    start_time = now_utc()

    # Validate server exists
    if server not in MCP_SERVERS:
        raise HTTPException(status_code=404, detail=f"Server '{server}' not found")

    # Check permissions
    if not check_permission(user, server, method):
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions for {server}:{method}"
        )

    # Rate limiting
    if not await rate_limiter.check_rate_limit(user.username, f"{server}:{method}"):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

    # Get client IP
    client_ip = req.client.host if req.client else "unknown"

    # Audit logging
    await audit_logger.log_request(
        user.username,
        server,
        method,
        request.params,
        client_ip
    )

    # Forward request to MCP server
    server_url = MCP_SERVERS[server]["url"]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server_url}/mcp/{method}",
                json=request.dict(),
                headers={
                    "X-User-ID": user.username,
                    "X-Request-ID": f"{now_utc().timestamp()}",
                    "X-Forwarded-For": client_ip
                },
                timeout=30.0
            )

        # Calculate latency
        latency_ms = (now_utc() - start_time).total_seconds() * 1000

        # Audit response
        await audit_logger.log_response(
            user.username,
            server,
            method,
            response.status_code == 200,
            latency_ms
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        # Return response with metadata
        result = response.json()
        return MCPResponse(
            result=result,
            metadata={
                "server": server,
                "method": method,
                "latency_ms": round(latency_ms, 2),
                "timestamp": now_utc().isoformat()
            }
        )

    except httpx.RequestError as e:
        # Log error
        await audit_logger.log_response(
            user.username,
            server,
            method,
            False,
            (now_utc() - start_time).total_seconds() * 1000
        )

        raise HTTPException(
            status_code=503,
            detail=f"Server '{server}' is temporarily unavailable: {str(e)}"
        )

@app.get("/mcp/servers")
async def list_servers(user: TokenData = Depends(verify_token)):
    """List available MCP servers and their capabilities"""
    servers = {}

    for server_name, server_info in MCP_SERVERS.items():
        # Check if user has any permission for this server
        if any(scope.startswith(f"{server_name}:") for scope in user.scopes) or "admin" in user.scopes:
            servers[server_name] = {
                "description": server_info["description"],
                "status": "available",  # TODO: Real-time status check
                "permissions": [s for s in user.scopes if s.startswith(f"{server_name}:")]
            }

    return {
        "servers": servers,
        "user": user.username,
        "timestamp": now_utc().isoformat()
    }

@app.get("/audit/logs")
async def get_audit_logs(
    date: Optional[str] = None,
    limit: int = 100,
    user: TokenData = Depends(verify_token)
):
    """Get audit logs (admin only)"""
    if "admin" not in user.scopes:
        raise HTTPException(status_code=403, detail="Admin access required")

    # Use today's date if not specified
    if date is None:
        date = now_utc().strftime("%Y%m%d")

    # Get logs from Redis
    key = f"audit:{date}:*"
    logs = []

    # TODO: Implement proper log retrieval
    # For now, return empty list

    return {
        "logs": logs,
        "date": date,
        "count": len(logs),
        "limit": limit
    }

@app.get("/metrics")
async def get_metrics(user: TokenData = Depends(verify_token)):
    """Get gateway metrics"""
    if "admin" not in user.scopes:
        raise HTTPException(status_code=403, detail="Admin access required")

    # TODO: Implement proper metrics collection
    return {
        "requests_total": 0,
        "requests_per_minute": 0,
        "average_latency_ms": 0,
        "error_rate": 0,
        "active_users": 0,
        "timestamp": now_utc().isoformat()
    }

if __name__ == "__main__":
    print("Starting MCP Gateway Server")
    print("Available servers:", list(MCP_SERVERS.keys()))
    print("API docs available at http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
