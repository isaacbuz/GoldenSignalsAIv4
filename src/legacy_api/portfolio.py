from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

# Mock portfolio state (would be replaced by DB or live engine)
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

@router.get("/api/portfolio/overview")
def portfolio_overview():
    return PORTFOLIO
