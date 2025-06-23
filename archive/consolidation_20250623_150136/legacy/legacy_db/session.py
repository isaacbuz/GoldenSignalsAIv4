from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

from src.legacy_config.models import Base

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_session():
    return SessionLocal()
