"""
Script to initialize (create) all tables in the AWS RDS PostgreSQL database for GoldenSignalsAI.
Usage:
  conda run -n goldensignalsai python scripts/init_db.py
"""
from domain.models.notification_preferences import Base
from infrastructure.data.db_session import engine

if __name__ == "__main__":
    print("Creating all tables in the database...")
    Base.metadata.create_all(engine)
    print("Done. Database schema is ready.")
