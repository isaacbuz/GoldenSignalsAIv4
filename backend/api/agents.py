from fastapi import APIRouter, Query, Request, HTTPException
from backend.agents.agent_manager import ALL_AGENTS, AgentManager
from backend.db.agent_config import load_agent_config, save_agent_config
from backend.db.retrain_logs import append_retrain_log, get_retrain_logs
from backend.api.ws import broadcast_feedback_event, broadcast_agent_event
from typing import List, Dict, Any
import uuid
import datetime
import logging

router = APIRouter()

# In-memory feedback store for demo; replace with DB in production
FEEDBACK_STORE: List[Dict[str, Any]] = []

@router.get("/signals")
async def get_agent_signals(symbol: str = Query(..., example="AAPL")):
    agents = [AgentClass(symbol) for AgentClass in ALL_AGENTS]
    manager = AgentManager(agents)
    signals = manager.run_all()
    consensus = manager.vote()
    return {"signals": signals, "consensus": consensus}

@router.post("/feedback")
async def submit_feedback(feedback: Dict):
    # Expected: symbol, agent, rating, comment
    entry = dict(feedback)
    entry["id"] = str(uuid.uuid4())
    entry["timestamp"] = datetime.datetime.utcnow().isoformat()
    FEEDBACK_STORE.append(entry)
    return {"success": True, "id": entry["id"]}

@router.get("/feedback")
async def get_feedback():
    # Admin: view all feedback
    return {"feedback": FEEDBACK_STORE}

# --- Advanced agent logic/admin endpoints (automated) ---
import subprocess

@router.post("/admin/retrain")
async def retrain_agent(data: Dict):
    agent = data.get("agent")
    # Example: Only ML agents are retrainable
    ml_agents = ["MLTrendChannelAgent", "ForecastAgent", "SentimentAgent"]
    if agent in ml_agents:
        try:
            result = subprocess.run([
                "python", "backend/ml_training/retrain_all.py", "--agent", agent
            ], capture_output=True, text=True, timeout=600)
            status = "retrain complete" if result.returncode == 0 else "retrain failed"
            output = result.stdout + result.stderr
            append_retrain_log(agent, status, output)
            await broadcast_agent_event({
                "type": "retrain",
                "agent": agent,
                "status": status,
                "output": output,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            })
            return {
                "status": status,
                "agent": agent,
                "output": output,
                "code": result.returncode
            }
        except Exception as e:
            logging.error(f"Retrain error: {e}")
            append_retrain_log(agent, "error", str(e))
            await broadcast_agent_event({
                "type": "retrain",
                "agent": agent,
                "status": "error",
                "output": str(e),
                "timestamp": datetime.datetime.utcnow().isoformat(),
            })
            return {"status": "error", "agent": agent, "error": str(e)}
    else:
        logging.info(f"Retrain not supported for agent: {agent}")
        append_retrain_log(agent, "not supported", "")
        await broadcast_agent_event({
            "type": "retrain",
            "agent": agent,
            "status": "not supported",
            "output": "",
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })
        return {"status": "not supported", "agent": agent}

@router.get("/admin/retrain_logs")
async def fetch_retrain_logs(agent: str = None):
    logs = get_retrain_logs(agent)
    return {"logs": logs}

@router.get("/admin/config")
async def get_agent_config(agent: str):
    config = load_agent_config(agent)
    return {"agent": agent, "config": config}

@router.post("/admin/config")
async def update_agent_config(data: Dict):
    agent = data.get("agent")
    config = data.get("config", {})
    save_agent_config(agent, config)
    return {"status": "config updated", "agent": agent, "config": config}
