from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr
from typing import Optional
from infrastructure.auth.jwt_utils import verify_jwt_token
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr
from typing import Optional
from infrastructure.auth.jwt_utils import verify_jwt_token
from infrastructure.data.db_session import get_db
from sqlalchemy.orm import Session
from domain.models.notification_preferences import NotificationPreferences

router = APIRouter()

class NotificationSettings(BaseModel):
    slack: Optional[str] = None
    discord: Optional[str] = None
    email: Optional[EmailStr] = None
    sms: Optional[str] = None
    push: Optional[bool] = False
    highConfidenceOnly: Optional[bool] = True

@router.post('/user/notifications', status_code=200)
async def set_notification_settings(settings: NotificationSettings, user=Depends(verify_jwt_token), db: Session = Depends(get_db)):
    prefs = db.query(NotificationPreferences).filter_by(username=user.username).first()
    if prefs:
        prefs.slack = settings.slack
        prefs.discord = settings.discord
        prefs.email = settings.email
        prefs.sms = settings.sms
        prefs.push = settings.push
        prefs.high_confidence_only = settings.highConfidenceOnly
    else:
        prefs = NotificationPreferences(
            username=user.username,
            slack=settings.slack,
            discord=settings.discord,
            email=settings.email,
            sms=settings.sms,
            push=settings.push,
            high_confidence_only=settings.highConfidenceOnly
        )
        db.add(prefs)
    db.commit()
    return {"status": "success"}

@router.get('/user/notifications', response_model=NotificationSettings)
async def get_notification_settings(user=Depends(verify_jwt_token), db: Session = Depends(get_db)):
    prefs = db.query(NotificationPreferences).filter_by(username=user.username).first()
    if not prefs:
        return NotificationSettings()
    return NotificationSettings(
        slack=prefs.slack,
        discord=prefs.discord,
        email=prefs.email,
        sms=prefs.sms,
        push=prefs.push,
        highConfidenceOnly=prefs.high_confidence_only
    )
