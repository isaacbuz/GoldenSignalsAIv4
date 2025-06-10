from fastapi import APIRouter, Query
from datetime import datetime, timedelta
import random

router = APIRouter()

@router.get("/api/options/flow")
def get_options_flow(symbol: str = Query(..., description="Stock symbol")):
    # Generate mock options trades
    now = datetime.utcnow()
    trades = []

    for _ in range(40):
        trade = {
            "type": random.choice(["call", "put"]),
            "direction": "buy",
            "strike": random.randint(120, 160),
            "iv": round(random.uniform(0.2, 0.5), 2),
            "size": random.randint(200, 2000),
            "timestamp": (now - timedelta(minutes=random.randint(1, 120))).isoformat()
        }
        trades.append(trade)

    return {"symbol": symbol.upper(), "trades": trades}
