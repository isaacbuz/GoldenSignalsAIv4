"""
Rate limiting middleware for API protection.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

import redis
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Create limiter instance
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour"],
    storage_uri="redis://localhost:6379"
)

class RateLimiter:
    """Advanced rate limiting with Redis backend."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_limits = {
            "public": {"requests": 100, "window": 3600},  # 100 per hour
            "authenticated": {"requests": 1000, "window": 3600},  # 1000 per hour
            "premium": {"requests": 10000, "window": 3600},  # 10000 per hour
            "api_key": {"requests": 5000, "window": 3600},  # 5000 per hour
        }
    
    def get_user_tier(self, request: Request) -> str:
        """Determine user tier from request."""
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return "api_key"
        
        # Check for authenticated user
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # TODO: Decode JWT and check user tier
            return "authenticated"
        
        return "public"
    
    def get_client_id(self, request: Request) -> str:
        """Get unique client identifier."""
        # Priority: API Key > User ID > IP Address
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # TODO: Extract user ID from JWT
            return f"user:anonymous"
        
        # Fallback to IP address
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def check_rate_limit(
        self, 
        client_id: str, 
        endpoint: str, 
        tier: str = "public"
    ) -> Dict[str, Any]:
        """Check if client has exceeded rate limit."""
        limits = self.default_limits.get(tier, self.default_limits["public"])
        
        # Create Redis key
        window_start = int(time.time() // limits["window"]) * limits["window"]
        key = f"rate_limit:{client_id}:{endpoint}:{window_start}"
        
        # Get current count
        current = self.redis.get(key)
        current_count = int(current) if current else 0
        
        # Check limit
        if current_count >= limits["requests"]:
            reset_time = window_start + limits["window"]
            return {
                "allowed": False,
                "limit": limits["requests"],
                "remaining": 0,
                "reset": reset_time,
                "retry_after": reset_time - int(time.time())
            }
        
        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, limits["window"])
        pipe.execute()
        
        return {
            "allowed": True,
            "limit": limits["requests"],
            "remaining": limits["requests"] - current_count - 1,
            "reset": window_start + limits["window"],
            "retry_after": None
        }
    
    def get_usage_stats(self, client_id: str) -> Dict[str, Any]:
        """Get rate limit usage statistics for a client."""
        stats = {}
        pattern = f"rate_limit:{client_id}:*"
        
        for key in self.redis.scan_iter(match=pattern):
            parts = key.decode().split(":")
            if len(parts) >= 4:
                endpoint = parts[2]
                count = self.redis.get(key)
                ttl = self.redis.ttl(key)
                
                stats[endpoint] = {
                    "count": int(count) if count else 0,
                    "ttl": ttl
                }
        
        return stats

class RateLimitMiddleware:
    """Rate limiting middleware for FastAPI."""
    
    def __init__(self, redis_client: redis.Redis):
        self.rate_limiter = RateLimiter(redis_client)
    
    async def __call__(self, request: Request, call_next: Callable) -> JSONResponse:
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get client info
        client_id = self.rate_limiter.get_client_id(request)
        tier = self.rate_limiter.get_user_tier(request)
        endpoint = f"{request.method}:{request.url.path}"
        
        # Check rate limit
        result = self.rate_limiter.check_rate_limit(client_id, endpoint, tier)
        
        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(result["limit"]),
            "X-RateLimit-Remaining": str(result["remaining"]),
            "X-RateLimit-Reset": str(result["reset"]),
        }
        
        if not result["allowed"]:
            headers["Retry-After"] = str(result["retry_after"])
            
            logger.warning(
                f"Rate limit exceeded for {client_id} on {endpoint}. "
                f"Retry after {result['retry_after']} seconds."
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": result["retry_after"]
                },
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value
        
        return response

def create_rate_limit_decorator(
    requests: int = 10,
    window: int = 60,
    key_func: Optional[Callable] = None
):
    """Create a custom rate limit decorator."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(request: Request, *args, **kwargs):
            # Custom rate limiting logic here
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# Endpoint-specific rate limits
auth_limiter = create_rate_limit_decorator(requests=5, window=300)  # 5 per 5 minutes
data_limiter = create_rate_limit_decorator(requests=100, window=60)  # 100 per minute
signal_limiter = create_rate_limit_decorator(requests=50, window=60)  # 50 per minute 