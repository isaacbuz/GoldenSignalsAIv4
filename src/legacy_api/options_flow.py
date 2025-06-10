from fastapi import APIRouter, Query
import json
import os
from datetime import datetime

router = APIRouter()
MOCK_FILE = os.path.join(os.path.dirname(__file__), "options_flow_mock.json")

def filter_trades(trades, min_size, type_filter, sweep_only):
    filtered = [
        t for t in trades
        if t["size"] >= min_size and
           (type_filter == "all" or t["type"] == type_filter) and
           (not sweep_only or t["sweep"])
    ]
    return filtered

@router.get("/api/options_flow")
def get_options_flow(
    symbol: str = Query("AAPL"),
    minSize: int = Query(0),
    type: str = Query("all"),
    sweepOnly: bool = Query(False)
):
    # For now, always use mock data; later, fetch real data per symbol
    if os.path.exists(MOCK_FILE):
        with open(MOCK_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {"trades": [], "iv": []}
    trades = filter_trades(data.get("trades", []), minSize, type, sweepOnly)
    return {
        "trades": trades,
        "iv": data.get("iv", [])
    }
