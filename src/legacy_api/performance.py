from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.legacy_db.connect import SessionLocal
from src.legacy_db.models import Performance
from sqlalchemy import select
from datetime import datetime

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/api/performance")
def get_performance(symbol: str = None, db: Session = Depends(get_db)):
    q = select(Performance)
    if symbol:
        q = q.where(Performance.symbol == symbol)
    results = db.execute(q).scalars().all()
    return [
        {
            "id": p.id,
            "symbol": p.symbol,
            "date": p.date.isoformat(),
            "pnl": p.pnl,
            "return_pct": p.return_pct
        }
        for p in results
    ]

@router.post("/api/performance")
def create_performance(payload: dict, db: Session = Depends(get_db)):
    symbol = payload.get("symbol")
    pnl = payload.get("pnl")
    return_pct = payload.get("return_pct")
    date = payload.get("date", datetime.utcnow())
    if not symbol or pnl is None or return_pct is None:
        raise HTTPException(status_code=400, detail="Missing required fields")
    perf = Performance(symbol=symbol, pnl=pnl, return_pct=return_pct, date=date)
    db.add(perf)
    db.commit()
    db.refresh(perf)
    return {
        "id": perf.id,
        "symbol": perf.symbol,
        "date": perf.date.isoformat(),
        "pnl": perf.pnl,
        "return_pct": perf.return_pct
    }
