"""
Dependency Injection for GoldenSignalsAI V3

FastAPI dependency providers for services, authentication, and shared resources.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from src.services.market_data_service import MarketDataService
from src.services.signal_service import SignalService

from .config import settings
from .database import DatabaseManager
from .redis_manager import RedisManager

# Global service instances (initialized in main.py lifespan)
_db_manager: Optional[DatabaseManager] = None
_redis_manager: Optional[RedisManager] = None
_signal_service: Optional[SignalService] = None
_market_data_service: Optional[MarketDataService] = None

# Security scheme
security = HTTPBearer(auto_error=False)


def set_global_services(
    db_manager: DatabaseManager,
    redis_manager: RedisManager,
    signal_service: SignalService,
    market_data_service: MarketDataService,
) -> None:
    """
    Set global service instances (called from main.py during startup)
    """
    global _db_manager, _redis_manager, _signal_service, _market_data_service
    _db_manager = db_manager
    _redis_manager = redis_manager
    _signal_service = signal_service
    _market_data_service = market_data_service


async def get_db_manager() -> DatabaseManager:
    """Get database manager dependency"""
    if not _db_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database manager not initialized",
        )
    return _db_manager


async def get_redis_manager() -> RedisManager:
    """Get Redis manager dependency"""
    if not _redis_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Redis manager not initialized",
        )
    return _redis_manager


async def get_signal_service() -> SignalService:
    """Get signal service dependency"""
    if not _signal_service:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signal service not initialized",
        )
    return _signal_service


async def get_market_data_service() -> MarketDataService:
    """Get market data service dependency"""
    if not _market_data_service:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Market data service not initialized",
        )
    return _market_data_service


async def get_agent_orchestrator():
    """Get agent orchestrator dependency"""
    # This would be injected from the main application
    # For now, return a mock or raise an error
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Agent orchestrator not available in this context",
    )


def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token and return payload

    Args:
        token: JWT token string

    Returns:
        Dict containing token payload

    Raises:
        HTTPException: If token is invalid
    """
    try:
        payload = jwt.decode(
            token, settings.security.secret_key, algorithms=[settings.security.algorithm]
        )

        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return payload

    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Dict containing user information

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        # For development, allow unauthenticated access
        if settings.debug:
            return {
                "user_id": "dev_user",
                "username": "developer",
                "roles": ["user", "admin"],
                "is_authenticated": False,
            }

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_jwt_token(credentials.credentials)

    return {
        "user_id": payload.get("sub"),
        "username": payload.get("username"),
        "email": payload.get("email"),
        "roles": payload.get("roles", ["user"]),
        "is_authenticated": True,
        "token_payload": payload,
    }


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, otherwise return None

    Args:
        credentials: HTTP authorization credentials

    Returns:
        User dict if authenticated, None otherwise
    """
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_roles(required_roles: list):
    """
    Dependency factory for role-based access control

    Args:
        required_roles: List of required roles

    Returns:
        Dependency function
    """

    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_roles = current_user.get("roles", [])

        if not any(role in user_roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {required_roles}",
            )

        return current_user

    return role_checker


async def check_rate_limit(
    request: Request, redis_manager: RedisManager = Depends(get_redis_manager)
) -> None:
    """
    Check rate limiting for API requests

    Args:
        request: FastAPI request object
        redis_manager: Redis manager for rate limiting

    Raises:
        HTTPException: If rate limit exceeded
    """
    # Get client identifier (IP address or user ID)
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")
    identifier = f"{client_ip}:{user_agent}"

    # Check rate limit
    within_limit = await redis_manager.check_rate_limit(
        identifier=identifier,
        max_requests=settings.security.rate_limit_requests,
        window_seconds=settings.security.rate_limit_window,
    )

    if not within_limit:
        logger.warning(f"Rate limit exceeded for {identifier}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(settings.security.rate_limit_window)},
        )


async def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize stock symbol

    Args:
        symbol: Raw symbol string

    Returns:
        Normalized symbol

    Raises:
        HTTPException: If symbol is invalid
    """
    symbol = symbol.upper().strip()

    # Basic validation
    if not symbol or len(symbol) > 10 or not symbol.isalpha():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid symbol format")

    return symbol


async def get_request_context(request: Request) -> Dict[str, Any]:
    """
    Get request context information for logging and tracking

    Args:
        request: FastAPI request object

    Returns:
        Dict containing request context
    """
    return {
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent"),
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request.headers.get("x-request-id", "unknown"),
    }


class PaginationParams:
    """
    Pagination parameters for API endpoints
    """

    def __init__(self, page: int = 1, page_size: int = 20, max_page_size: int = 100):
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Page number must be >= 1"
            )

        if page_size < 1 or page_size > max_page_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Page size must be between 1 and {max_page_size}",
            )

        self.page = page
        self.page_size = page_size
        self.offset = (page - 1) * page_size


def get_pagination_params(page: int = 1, page_size: int = 20) -> PaginationParams:
    """
    Get pagination parameters with validation

    Args:
        page: Page number
        page_size: Items per page

    Returns:
        PaginationParams object
    """
    return PaginationParams(page, page_size)
