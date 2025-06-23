from fastapi import APIRouter
import pandas as pd
import os

router = APIRouter()

@router.get("/api/analytics/agent_accuracy")
def agent_accuracy():
    from archive.legacy_backend_agents.edge.agent_memory_manager import AgentMemoryManager
    memory = AgentMemoryManager()
    return {agent: memory.get_accuracy(agent) for agent in memory.memory}

@router.get("/api/analytics/trade_performance")
def trade_performance():
    trade_log_path = os.path.join(os.path.dirname(__file__), "..", "data", "trade_log.csv")
    if not os.path.exists(trade_log_path):
        return {"error": "No trade log found."}
    df = pd.read_csv(trade_log_path)
    return {
        "total_trades": int(len(df)),
        "win_rate": float((df["return_pct"] > 0).mean()),
        "avg_return": float(df["return_pct"].mean()),
        "max_drawdown": float(df["return_pct"].cumsum().min()),
    }

@router.get("/api/analytics/portfolio_curve")
def portfolio_curve():
    trade_log_path = os.path.join(os.path.dirname(__file__), "..", "data", "trade_log.csv")
    if not os.path.exists(trade_log_path):
        return {"error": "No trade log found."}
    df = pd.read_csv(trade_log_path)
    df["equity_curve"] = (1 + df["return_pct"]).cumprod()
    return {
        "dates": df["entry_date"].tolist(),
        "equity_curve": df["equity_curve"].tolist(),
    }
