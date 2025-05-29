# Placeholder FastAPI endpoints for missing frontend routes
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.db.connect import SessionLocal
from backend.db.models import SignalRecord

# --- Real DB-backed Watchlist Endpoints ---
from sqlalchemy import select, insert, delete

WATCHLIST_TABLE = "watchlist"

# Ensure the table exists (for demo, should be in models.py)
from sqlalchemy import Table, Column, Integer, String, MetaData
metadata = MetaData()
watchlist_table = Table(
    WATCHLIST_TABLE, metadata,
    Column("id", Integer, primary_key=True),
    Column("symbol", String, unique=True, index=True)
)
engine = SessionLocal().get_bind()
metadata.create_all(engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/api/watchlist")
def get_watchlist(db: Session = Depends(get_db)):
    result = db.execute(select(watchlist_table.c.symbol)).fetchall()
    symbols = [row[0] for row in result]
    return JSONResponse({"symbols": symbols})

@router.post("/api/watchlist")
def add_watchlist_symbol(payload: dict, db: Session = Depends(get_db)):
    symbol = payload.get("symbol", "").upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol required")
    exists = db.execute(select(watchlist_table.c.symbol).where(watchlist_table.c.symbol == symbol)).fetchone()
    if not exists:
        db.execute(insert(watchlist_table).values(symbol=symbol))
        db.commit()
    return get_watchlist(db)

@router.delete("/api/watchlist")
def remove_watchlist_symbol(payload: dict, db: Session = Depends(get_db)):
    symbol = payload.get("symbol", "").upper()
    db.execute(delete(watchlist_table).where(watchlist_table.c.symbol == symbol))
    db.commit()
    return get_watchlist(db)

# Alerts endpoints are now handled by api/alerts.py and included in app.py

@router.get("/api/performance")
def get_performance():
    return JSONResponse({"performance": []})

@router.get("/api/price-history")
def get_price_history(symbol: str = "AAPL"):
    return JSONResponse({"symbol": symbol, "history": []})

@router.get("/api/logs")
def get_logs():
    return JSONResponse([{"id": 1, "log": "No logs yet."}])

@router.get("/api/confidence-heatmap")
def get_confidence_heatmap():
    return JSONResponse({"heatmap": []})

@router.get("/api/tv-comparison")
def get_tv_comparison():
    return JSONResponse({"comparison": []})

@router.websocket("/ws/signals")
async def websocket_signals(websocket):
    await websocket.accept()
    import asyncio
    import json
    # Send dummy signal every 5 seconds
    while True:
        await websocket.send_text(json.dumps({
            "id": 1,
            "symbol": "AAPL",
            "action": "BUY",
            "time": "09:30:00",
            "confidence": 95,
            "reason": "Test signal"
        }))
        await asyncio.sleep(5)
