from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

# Mock trade log (replace with DB or persistent store in production)
TRADE_LOG = [
    {
        "id": 1,
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": "AAPL",
        "side": "buy",
        "qty": 100,
        "price": 150.0,
        "agent": "BreakoutAgent",
        "pnl": 420,
        "attribution": ["BreakoutAgent", "RSIAgent"],
        "risk_alert": False,
    },
    {
        "id": 2,
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": "TSLA",
        "side": "sell",
        "qty": 50,
        "price": 700.0,
        "agent": "SentimentAgent",
        "pnl": -225,
        "attribution": ["SentimentAgent"],
        "risk_alert": True,
    },
]

@router.get("/api/portfolio/trades")
def get_trades():
    return {"trades": TRADE_LOG}
