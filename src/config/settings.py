
"""Mock settings module for testing"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "GoldenSignalsAI"
    debug: bool = True
    database_url: str = "postgresql://test:test@localhost/test"
    redis_url: str = "redis://localhost:6379"
    
settings = Settings()
