from fastapi import APIRouter
import random

router = APIRouter()

@router.get("/api/signals/cluster-stats")
def signal_clusters():
    regimes = ["normal", "high_volatility", "low_volatility"]
    signals = ["bullish", "bearish", "neutral"]
    data = []
    for _ in range(5):
        data.append({
            "regime": random.choice(regimes),
            "signal": random.choice(signals),
            "avg_conf": round(random.uniform(0.55, 0.92), 2)
        })
    return data
