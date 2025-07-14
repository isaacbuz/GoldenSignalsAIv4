"""
API Rate Limiting Middleware
Protects the API from abuse and ensures fair usage
"""

import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import redis
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib
import json

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-based rate limiter using sliding window algorithm"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.enabled = redis_client is not None
    
    def _get_identifier(self, request: Request) -> str:
        """
        Get unique identifier for the request
        Priority: API Key > User ID > IP Address
        """
        # Check for API key
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"api_key:{hashlib.md5(api_key.encode()).hexdigest()}"
        
        # Check for authenticated user
        if hasattr(request.state, 'user') and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            client_ip = forwarded_for.split(',')[0].strip()
        else:
            client_ip = request.client.host if request.client else 'unknown'
        
        return f"ip:{client_ip}"
    
    def _get_rate_limit_key(self, identifier: str, endpoint: str) -> str:
        """Generate Redis key for rate limiting"""
        return f"rate_limit:{identifier}:{endpoint}"
    
    async def check_rate_limit(
        self,
        request: Request,
        limit: int,
        window_seconds: int
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is within rate limit
        
        Args:
            request: FastAPI request object
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (allowed, metadata)
        """
        if not self.enabled:
            return True, {}
        
        identifier = self._get_identifier(request)
        endpoint = request.url.path
        key = self._get_rate_limit_key(identifier, endpoint)
        
        now = time.time()
        window_start = now - window_seconds
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count requests in current window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiry
            pipe.expire(key, window_seconds + 1)
            
            # Execute pipeline
            results = pipe.execute()
            
            request_count = results[1]
            
            # Check if limit exceeded
            if request_count >= limit:
                # Get oldest request time for retry-after calculation
                oldest = self.redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(window_seconds - (now - oldest[0][1]))
                else:
                    retry_after = window_seconds
                
                return False, {
                    'limit': limit,
                    'remaining': 0,
                    'reset': int(now + retry_after),
                    'retry_after': retry_after
                }
            
            return True, {
                'limit': limit,
                'remaining': limit - request_count - 1,
                'reset': int(now + window_seconds)
            }
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if Redis fails
            return True, {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""
    
    # Default rate limits per endpoint pattern
    DEFAULT_LIMITS = {
        '/api/v1/signals/generate': (10, 60),      # 10 requests per minute
        '/api/v1/signals': (100, 60),              # 100 requests per minute
        '/api/v1/market-data': (300, 60),          # 300 requests per minute
        '/ws': (5, 60),                            # 5 WebSocket connections per minute
        'default': (200, 60)                       # 200 requests per minute default
    }
    
    # Premium user limits (higher)
    PREMIUM_LIMITS = {
        '/api/v1/signals/generate': (50, 60),      # 50 requests per minute
        '/api/v1/signals': (500, 60),              # 500 requests per minute
        '/api/v1/market-data': (1000, 60),         # 1000 requests per minute
        '/ws': (20, 60),                           # 20 WebSocket connections per minute
        'default': (1000, 60)                      # 1000 requests per minute default
    }
    
    def __init__(self, app, redis_url: Optional[str] = None):
        super().__init__(app)
        
        # Initialize Redis connection
        try:
            if redis_url:
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                self.redis_client.ping()
                self.rate_limiter = RateLimiter(self.redis_client)
                logger.info("✅ Rate limiting enabled with Redis")
            else:
                self.redis_client = None
                self.rate_limiter = None
                logger.warning("⚠️ Rate limiting disabled (no Redis URL)")
        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {e}")
            self.redis_client = None
            self.rate_limiter = None
    
    def _get_limits_for_endpoint(self, path: str, is_premium: bool = False) -> Tuple[int, int]:
        """Get rate limits for specific endpoint"""
        limits = self.PREMIUM_LIMITS if is_premium else self.DEFAULT_LIMITS
        
        # Find matching pattern
        for pattern, limit in limits.items():
            if pattern == 'default':
                continue
            if path.startswith(pattern):
                return limit
        
        return limits['default']
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        
        # Skip rate limiting for health checks and docs
        if request.url.path in ['/', '/health', '/docs', '/redoc', '/openapi.json']:
            return await call_next(request)
        
        # Skip if rate limiting is disabled
        if not self.rate_limiter:
            return await call_next(request)
        
        # Check if user is premium (from auth middleware)
        is_premium = False
        if hasattr(request.state, 'user') and request.state.user:
            is_premium = getattr(request.state.user, 'is_premium', False)
        
        # Get limits for endpoint
        limit, window = self._get_limits_for_endpoint(request.url.path, is_premium)
        
        # Check rate limit
        allowed, metadata = await self.rate_limiter.check_rate_limit(
            request, limit, window
        )
        
        if not allowed:
            # Return 429 Too Many Requests
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Please retry after {metadata.get("retry_after", 60)} seconds.',
                    'limit': metadata.get('limit'),
                    'retry_after': metadata.get('retry_after')
                },
                headers={
                    'X-RateLimit-Limit': str(metadata.get('limit', limit)),
                    'X-RateLimit-Remaining': '0',
                    'X-RateLimit-Reset': str(metadata.get('reset', 0)),
                    'Retry-After': str(metadata.get('retry_after', 60))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers['X-RateLimit-Limit'] = str(metadata.get('limit', limit))
        response.headers['X-RateLimit-Remaining'] = str(metadata.get('remaining', 0))
        response.headers['X-RateLimit-Reset'] = str(metadata.get('reset', 0))
        
        return response


def create_rate_limit_decorator(limit: int = 100, window: int = 60):
    """
    Create a decorator for custom rate limits on specific endpoints
    
    Args:
        limit: Maximum requests allowed
        window: Time window in seconds
    """
    def decorator(func):
        func._rate_limit = (limit, window)
        return func
    return decorator


# Convenience decorators
rate_limit_low = create_rate_limit_decorator(10, 60)      # 10/minute
rate_limit_medium = create_rate_limit_decorator(50, 60)    # 50/minute
rate_limit_high = create_rate_limit_decorator(200, 60)     # 200/minute