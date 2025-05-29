from sqlalchemy import Column, Integer, String, DateTime, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class AgentConfig(Base):
    __tablename__ = 'agent_configs'
    id = Column(Integer, primary_key=True)
    agent = Column(String, unique=True, index=True)
    config = Column(JSON)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

class FeedbackEntry(Base):
    __tablename__ = 'feedback_entries'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    agent = Column(String)
    action = Column(String)
    rating = Column(Float)
    comment = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class RetrainLog(Base):
    __tablename__ = 'retrain_logs'
    id = Column(Integer, primary_key=True)
    agent = Column(String)
    status = Column(String)
    output = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
