"""
Agents API Endpoints - GoldenSignalsAI V3

REST API endpoints for managing and monitoring AI trading agents.
"""

import logging
from typing import Any, Dict, List

from agents.orchestrator import AgentOrchestrator
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.core.dependencies import get_agent_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()


class AgentStatus(BaseModel):
    """Agent status response model"""
    name: str
    type: str
    status: str
    accuracy: float
    total_signals: int
    correct_signals: int
    avg_confidence: float
    last_signal_time: str
    performance_metrics: Dict[str, Any]


class OrchestratorStatus(BaseModel):
    """Orchestrator status response model"""
    is_running: bool
    active_agents: int
    total_signals_generated: int
    consensus_threshold: float
    signal_generation_rate: float
    system_health: str


@router.get("/", response_model=List[AgentStatus])
async def get_agents(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> List[AgentStatus]:
    """
    Get list of all agents with their current status and performance metrics.
    """
    try:
        agents_data = await orchestrator.get_agents_status()
        
        agents = []
        for agent_name, data in agents_data.items():
            agent_status = AgentStatus(
                name=agent_name,
                type=data.get("type", "unknown"),
                status=data.get("status", "unknown"),
                accuracy=data.get("accuracy", 0.0),
                total_signals=data.get("total_signals", 0),
                correct_signals=data.get("correct_signals", 0),
                avg_confidence=data.get("avg_confidence", 0.0),
                last_signal_time=data.get("last_signal_time", ""),
                performance_metrics=data.get("performance_metrics", {})
            )
            agents.append(agent_status)
        
        return agents
        
    except Exception as e:
        logger.error(f"Error getting agents status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agents status"
        )


@router.get("/performance", response_model=Dict[str, Any])
async def get_agent_performance(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> Dict[str, Any]:
    """
    Get detailed performance metrics for all agents.
    """
    try:
        performance_data = await orchestrator.get_performance_metrics()
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting agent performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent performance metrics"
        )


@router.get("/orchestrator/status", response_model=OrchestratorStatus)
async def get_orchestrator_status(
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> OrchestratorStatus:
    """
    Get the current status of the agent orchestrator.
    """
    try:
        status_data = await orchestrator.get_orchestrator_status()
        
        return OrchestratorStatus(
            is_running=status_data.get("is_running", False),
            active_agents=status_data.get("active_agents", 0),
            total_signals_generated=status_data.get("total_signals_generated", 0),
            consensus_threshold=status_data.get("consensus_threshold", 0.0),
            signal_generation_rate=status_data.get("signal_generation_rate", 0.0),
            system_health=status_data.get("system_health", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve orchestrator status"
        )


@router.get("/{agent_name}/status")
async def get_agent_status(
    agent_name: str,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> Dict[str, Any]:
    """
    Get detailed status for a specific agent.
    """
    try:
        agent_status = await orchestrator.get_agent_status(agent_name)
        
        if not agent_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found"
            )
        
        return agent_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {agent_name} status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve status for agent '{agent_name}'"
        )


@router.post("/{agent_name}/restart")
async def restart_agent(
    agent_name: str,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
) -> Dict[str, str]:
    """
    Restart a specific agent.
    """
    try:
        success = await orchestrator.restart_agent(agent_name)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found or could not be restarted"
            )
        
        return {"message": f"Agent '{agent_name}' restarted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting agent {agent_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart agent '{agent_name}'"
        ) 