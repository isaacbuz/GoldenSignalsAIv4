"""
API V1 Router - GoldenSignalsAI V3

Main router that combines all API endpoints for version 1.
"""

from fastapi import APIRouter

from .agents import router as agents_router
from .analytics import router as analytics_router
from .backtesting import router as backtesting_router
from .market_data import router as market_data_router
from .notifications import router as notifications_router
from .portfolio import router as portfolio_router
from .signals import router as signals_router
from .integrated_signals import router as integrated_signals_router
from .websocket import router as websocket_router
from .ai_chat import router as ai_chat_router
from .ai_chat_enhanced import router as ai_chat_enhanced_router
from .hybrid_signals import router as hybrid_signals_router
from .auth import router as auth_router
from .admin import router as admin_router

# Create main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(
    agents_router,
    prefix="/agents",
    tags=["agents"]
)

api_router.include_router(
    analytics_router,
    prefix="/analytics",
    tags=["analytics"]
)

api_router.include_router(
    backtesting_router,
    prefix="/backtesting",
    tags=["backtesting"]
)

api_router.include_router(
    market_data_router,
    prefix="/market-data",
    tags=["market-data"]
)

api_router.include_router(
    notifications_router,
    prefix="/notifications",
    tags=["notifications"]
)

api_router.include_router(
    portfolio_router,
    prefix="/portfolio",
    tags=["portfolio"]
)

api_router.include_router(
    signals_router,
    prefix="/signals",
    tags=["signals"]
)

api_router.include_router(
    integrated_signals_router,
    prefix="/integrated-signals",
    tags=["integrated-signals"]
)

api_router.include_router(
    websocket_router,
    prefix="/ws",
    tags=["websocket"]
)

api_router.include_router(
    ai_chat_router,
    prefix="/ai-chat",
    tags=["ai-chat"]
)

api_router.include_router(
    ai_chat_enhanced_router,
    prefix="/ai-chat-enhanced",
    tags=["ai-chat-enhanced"]
)

api_router.include_router(
    hybrid_signals_router,
    prefix="/hybrid-signals",
    tags=["hybrid-signals"]
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

__all__ = ["api_router"] 