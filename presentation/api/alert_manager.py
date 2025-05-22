from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from infrastructure.auth.jwt_utils import verify_jwt_token
from application.services.alert_manager import AlertManager
import os

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])

EMAIL_CONFIG = {
    "sender": os.getenv("ALERT_EMAIL_SENDER", "demo@example.com"),
    "smtp_server": os.getenv("ALERT_SMTP_SERVER", "smtp.example.com"),
    "port": int(os.getenv("ALERT_SMTP_PORT", "465")),
    "password": os.getenv("ALERT_EMAIL_PASSWORD", "demo-password")
}
SLACK_WEBHOOK = os.getenv("ALERT_SLACK_WEBHOOK", "")
TELEGRAM_CONFIG = {
    "bot_token": os.getenv("ALERT_TELEGRAM_BOT_TOKEN", ""),
    "chat_id": os.getenv("ALERT_TELEGRAM_CHAT_ID", "")
}

alert_manager = AlertManager(email_config=EMAIL_CONFIG, slack_webhook=SLACK_WEBHOOK, telegram_config=TELEGRAM_CONFIG)

class AlertRequest(BaseModel):
    symbol: str
    channel: str  # email, slack, telegram
    threshold: float

class AlertResponse(BaseModel):
    status: str

@router.post("/subscribe")
async def subscribe_alert(request: AlertRequest, user=Depends(verify_jwt_token)):
    try:
        alert_manager.alert_all(request.symbol, f"Threshold alert for {request.symbol}", [request.channel])
        return {"status": "sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
