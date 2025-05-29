# API endpoint: /api/signal/aggregate
# Runs all backend indicator agents and returns unified signal output for a given market input
from backend.agents.signal_engine import run_all_agents
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/api/signal/aggregate")
async def aggregate_signals(request: Request):
    payload = await request.json()
    market_data = payload.get("market_data", {})
    symbol = payload.get("symbol", "UNKNOWN")
    result = run_all_agents(market_data, symbol)
    return JSONResponse(content=result)
