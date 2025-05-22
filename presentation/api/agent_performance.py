from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict
from infrastructure.auth.jwt_utils import verify_jwt_token
from application.monitoring.agent_performance_tracker import AgentPerformanceTracker

router = APIRouter(prefix="/api/v1/agent_performance", tags=["agent-performance"])

tracker = AgentPerformanceTracker()

class AgentPerformanceRequest(BaseModel):
    agent_id: str

class AgentPerformanceResponse(BaseModel):
    summary: Dict

@router.post("/summary", response_model=AgentPerformanceResponse)
async def agent_performance_summary(request: AgentPerformanceRequest, user=Depends(verify_jwt_token)):
    try:
        summary = tracker.get_summary(request.agent_id)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def agent_dashboard(user=Depends(verify_jwt_token)):
    try:
        agents = tracker.get_agents()
        agent_performance = []
        for agent in agents:
            summary = tracker.get_summary(agent["id"])
            agent_performance.append({
                "name": agent["name"],
                "win_rate": summary["win_rate"],
                "precision": summary["precision"],
                "recall": summary["recall"]
            })
        return {
            "agents": agent_performance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
