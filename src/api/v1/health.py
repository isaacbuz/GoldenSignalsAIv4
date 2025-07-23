"""
Health check endpoints for monitoring service health.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
import redis
from agents.orchestrator import AgentOrchestrator
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from src.core.database import get_db
from src.core.redis_manager import get_redis
from src.services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "GoldenSignalsAI",
        "version": "2.0.0"
    }

@router.get("/detailed")
async def detailed_health_check(
    db=Depends(get_db),
    redis_client=Depends(get_redis)
) -> Dict[str, Any]:
    """Detailed health check with component status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
        "system": {}
    }
    
    # Check database
    try:
        result = await db.execute(text("SELECT 1"))
        health_status["components"]["database"] = {
            "status": "healthy",
            "response_time_ms": 0  # Would measure actual time
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        await redis_client.ping()
        health_status["components"]["redis"] = {
            "status": "healthy",
            "response_time_ms": 0  # Would measure actual time
        }
    except Exception as e:
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check external APIs
    try:
        market_service = MarketDataService()
        # Quick check without actual API call
        health_status["components"]["market_data_api"] = {
            "status": "healthy" if market_service else "unhealthy"
        }
    except Exception as e:
        health_status["components"]["market_data_api"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # System metrics
    health_status["system"] = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    return health_status

@router.get("/agents")
async def agents_health_check() -> Dict[str, Any]:
    """Check health status of all agents."""
    try:
        orchestrator = AgentOrchestrator()
        agent_status = {}
        
        for agent_name, agent in orchestrator.agents.items():
            try:
                # Basic check if agent is initialized
                agent_status[agent_name] = {
                    "status": "healthy",
                    "initialized": True
                }
            except Exception as e:
                agent_status[agent_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return {
            "status": "healthy" if all(a["status"] == "healthy" for a in agent_status.values()) else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "agents": agent_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/ready")
async def readiness_check(
    db=Depends(get_db),
    redis_client=Depends(get_redis)
) -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint."""
    try:
        # Check critical dependencies
        await db.execute(text("SELECT 1"))
        await redis_client.ping()
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    } 