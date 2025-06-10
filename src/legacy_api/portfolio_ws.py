from fastapi import APIRouter, WebSocket
import asyncio
from datetime import datetime, timedelta

router = APIRouter()

# Mock data import (reuse from other API modules)
PORTFOLIO = {
    "equity": 105000,
    "cash": 25000,
    "leverage": 1.2,
    "drawdown": 0.032,
    "positions": [
        {"symbol": "AAPL", "size": 100, "entry_price": 150.0, "current_price": 154.2, "pnl": 420},
        {"symbol": "TSLA", "size": 50, "entry_price": 700.0, "current_price": 695.5, "pnl": -225},
    ]
}
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
RISK_ALERTS = [
    {
        "type": "Drawdown Limit",
        "message": "Portfolio drawdown exceeded 10% threshold!",
        "timestamp": (datetime.utcnow() - timedelta(minutes=2)).isoformat(),
    },
    {
        "type": "Exposure",
        "message": "AAPL position exceeds 30% of portfolio equity.",
        "timestamp": (datetime.utcnow() - timedelta(minutes=1)).isoformat(),
    },
]

@router.websocket("/ws/portfolio")
async def portfolio_ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        # In production, push on real update; here, just send every 3 seconds
        await websocket.send_json({
            "portfolio": PORTFOLIO,
            "trades": TRADE_LOG,
            "alerts": RISK_ALERTS,
            "attribution": TRADE_LOG,  # Attribution is part of trades for now
        })
        await asyncio.sleep(3)
