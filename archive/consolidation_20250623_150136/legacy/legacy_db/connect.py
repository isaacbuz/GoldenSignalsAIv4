from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.legacy_db.models import Base

DATABASE_URL = "postgresql://user:pass@localhost:5432/goldensignals"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
