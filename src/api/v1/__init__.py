"""
API V1 Router - GoldenSignalsAI V3

Main router that combines all API endpoints for version 1.
"""

from fastapi import APIRouter

from .signals import router as signals_router
from .agents import router as agents_router
from .market_data import router as market_data_router
from .portfolio import router as portfolio_router
from .auth import router as auth_router
from .admin import router as admin_router
from .analytics import router as analytics_router

# Create main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(
    signals_router,
    prefix="/signals",
    tags=["signals"]
)

api_router.include_router(
    agents_router,
    prefix="/agents",
    tags=["agents"]
)

api_router.include_router(
    market_data_router,
    prefix="/market-data",
    tags=["market-data"]
)

api_router.include_router(
    portfolio_router,
    prefix="/portfolio",
    tags=["portfolio"]
)

api_router.include_router(
    auth_router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    admin_router,
    prefix="/admin",
    tags=["admin"]
)

api_router.include_router(
    analytics_router,
    prefix="/analytics",
    tags=["analytics"]
) 