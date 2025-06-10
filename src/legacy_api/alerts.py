from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.legacy_db.connect import SessionLocal
from src.legacy_db.models import Alert
from sqlalchemy import select
from datetime import datetime

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/api/alerts")
def get_alerts(db: Session = Depends(get_db)):
    alerts = db.execute(select(Alert)).scalars().all()
    return [{"id": a.id, "message": a.message, "level": a.level, "timestamp": a.timestamp.isoformat()} for a in alerts]

@router.post("/api/alerts")
def create_alert(payload: dict, db: Session = Depends(get_db)):
    message = payload.get("message")
    level = payload.get("level", "info")
    if not message:
        raise HTTPException(status_code=400, detail="Message required")
    alert = Alert(message=message, level=level, timestamp=datetime.utcnow())
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return {"id": alert.id, "message": alert.message, "level": alert.level, "timestamp": alert.timestamp.isoformat()}
