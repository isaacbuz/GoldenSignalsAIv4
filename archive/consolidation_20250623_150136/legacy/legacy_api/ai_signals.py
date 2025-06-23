from fastapi import APIRouter, Request
from archive.legacy_backend_agents.agent_manager import AgentManager

router = APIRouter()

"""
POST /ai-signals
Body: { "market_data": {...}, "symbol": "AAPL", "use_tv": true }
Returns the output of all registered agents for the given symbol and market data.
"""
@router.post("/ai-signals")
async def get_ai_signals(req: Request):
    data = await req.json()
    market_data = data.get("market_data", {})
    symbol = data.get("symbol", "UNKNOWN")
    manager = AgentManager(symbol)
    signals = manager.run_all()
    consensus = manager.vote()
    return {"signals": signals, "consensus": consensus}
