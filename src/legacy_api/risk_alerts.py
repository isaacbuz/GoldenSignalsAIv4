from fastapi import APIRouter
from datetime import datetime, timedelta

router = APIRouter()

# Mock risk alerts (simulate what a real risk engine would output)
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

@router.get("/api/portfolio/risk_alerts")
def get_risk_alerts():
    return {"alerts": RISK_ALERTS}
