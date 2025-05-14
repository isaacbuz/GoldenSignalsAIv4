# admin_endpoints.py
# FastAPI endpoints for admin panel, protected by Firebase multi-auth
from fastapi import APIRouter, Depends
from .admin_auth import get_current_user, require_role
from .admin_metrics import get_metric_history, get_agent_health, get_queue_status

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.get("/performance")
async def performance(user=Depends(require_role("admin"))):
    # Dummy data: Replace with actual system metrics
    return {
        "cpu": 38.2,
        "memory": 71.5,
        "uptime": 123456,
        "requests": 129,
    }

alert_thresholds = {
    "memory": 70,
    "cpu": 90,
    "queue_depth": 100,
    # Add more as needed
}

@router.get("/alert_thresholds")
async def get_alert_thresholds(user=Depends(require_role("admin"))):
    return alert_thresholds

@router.post("/alert_thresholds")
async def set_alert_thresholds(new_thresholds: dict, user=Depends(require_role("admin"))):
    alert_thresholds.update(new_thresholds)
    return alert_thresholds

@router.get("/anomaly_check")
async def anomaly_check(user=Depends(require_role("admin"))):
    # Dummy anomaly detection logic
    anomalies = []
    # Example: Replace with real checks
    if 71.5 > alert_thresholds["memory"]:
        anomalies.append("High memory usage detected.")
    if 38.2 > alert_thresholds["cpu"]:
        anomalies.append("High CPU usage detected.")
    # Add more checks (e.g. agent health, queue depth)
    return {"alerts": anomalies}

@router.get("/performance/history")
async def get_performance_history(user=Depends(get_current_user)):
    return get_metric_history()

@router.get("/agents/health")
async def agent_health(user=Depends(get_current_user)):
    return get_agent_health()

@router.get("/queue")
async def queue_status(user=Depends(get_current_user)):
    return get_queue_status()

@router.get("/logs")
async def get_logs(user=Depends(get_current_user)):
    # Example: Read from your log file
    try:
        with open("./logs/goldensignalsai.log", "r") as f:
            return f.read()[-5000:]  # Last 5000 chars
    except Exception:
        return "No logs found."

@router.get("/agents")
async def list_agents(user=Depends(get_current_user)):
    # Example: Replace with real agent list
    return [
        {"name": "AlphaVantageAgent", "status": "active"},
        {"name": "FinnhubAgent", "status": "active"},
        {"name": "PolygonAgent", "status": "inactive"},
    ]

from .admin_audit import log_admin_action

@router.get("/agent_health")
async def agent_health(user=Depends(require_role("admin"))):
    # Dummy agent health data; replace with real agent status checks
    agents = [
        {
            "name": "AlphaVantageAgent",
            "status": "healthy",
            "last_heartbeat": "2025-05-14T19:59:00Z",
            "latency_ms": 120,
            "error_rate": 0.01
        },
        {
            "name": "FinnhubAgent",
            "status": "warning",
            "last_heartbeat": "2025-05-14T19:58:00Z",
            "latency_ms": 350,
            "error_rate": 0.07
        },
        {
            "name": "PolygonAgent",
            "status": "critical",
            "last_heartbeat": "2025-05-14T19:55:00Z",
            "latency_ms": 2000,
            "error_rate": 0.20
        }
    ]
    return {"agents": agents}

@router.get("/agents/{agent_name}")
async def agent_details(agent_name: str, user=Depends(get_current_user)):
    # Example: Replace with real agent details
    details = {
        "name": agent_name,
        "status": "active",
        "type": "DataSource",
        "lastActivity": "2025-05-14T19:00:00Z",
        "currentTask": "fetch_news",
        "successRate": 98.5,
        "errors": 1,
        "recentWork": ["Fetched price for AAPL", "Fetched news for MSFT"]
    }
    log_admin_action(user, "read_agent_details", target=agent_name)
    return details

from ..api.main import limiter

@router.post("/agents/{agent_name}/restart")
@limiter.limit("10/minute")
async def restart_agent(agent_name: str, user=Depends(require_role("admin"))):
    # Simulate agent restart with error handling and audit logging
    try:
        # Example: Check if agent exists (replace with real logic)
        valid_agents = ["AlphaVantageAgent", "FinnhubAgent", "PolygonAgent"]
        if agent_name not in valid_agents:
            log_admin_action(user, "restart_agent", target=agent_name, outcome="error", details="Agent not found")
            return {"success": False, "message": f"Agent {agent_name} not found."}
        # Simulate restart
        log_admin_action(user, "restart_agent", target=agent_name, outcome="success")
        return {"success": True, "message": f"Agent {agent_name} restarted."}
    except Exception as e:
        log_admin_action(user, "restart_agent", target=agent_name, outcome="error", details=str(e))
        return {"success": False, "message": f"Failed to restart {agent_name}: {str(e)}"}

@router.post("/agents/{agent_name}/disable")
@limiter.limit("10/minute")
async def disable_agent(agent_name: str, user=Depends(require_role("admin"))):
    # Simulate agent disable with error handling and audit logging
    try:
        valid_agents = ["AlphaVantageAgent", "FinnhubAgent", "PolygonAgent"]
        if agent_name not in valid_agents:
            log_admin_action(user, "disable_agent", target=agent_name, outcome="error", details="Agent not found")
            return {"success": False, "message": f"Agent {agent_name} not found."}
        # Simulate disable
        log_admin_action(user, "disable_agent", target=agent_name, outcome="success")
        return {"success": True, "message": f"Agent {agent_name} disabled."}
    except Exception as e:
        log_admin_action(user, "disable_agent", target=agent_name, outcome="error", details=str(e))
        return {"success": False, "message": f"Failed to disable {agent_name}: {str(e)}"}
