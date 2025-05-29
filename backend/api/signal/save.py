from fastapi import APIRouter, Request
from backend.agents.signal_engine import run_all_agents
from backend.db.save_signal import save_signals
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/api/signal/save")
async def save_signal_bundle(request: Request):
    payload = await request.json()
    market_data = payload.get("market_data", {})
    symbol = payload.get("symbol", "UNKNOWN")
    # Run all agents
    bundle = run_all_agents(market_data, symbol)
    # Save to DB
    save_signals(symbol, bundle["signals"], raw_data=market_data)
    return JSONResponse(content={"status": "ok", "bundle": bundle})
