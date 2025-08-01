from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class NotificationPreferences(Base):
    __tablename__ = "notification_preferences"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, index=True, nullable=False, unique=True)
    slack = Column(String, nullable=True)
    discord = Column(String, nullable=True)
    email = Column(String, nullable=True)
    sms = Column(String, nullable=True)
    push = Column(Boolean, default=False)
    high_confidence_only = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
