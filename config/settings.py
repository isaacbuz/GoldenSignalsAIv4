from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Application Configuration
    APP_NAME: str = "GoldenSignalsAI"
    APP_ENV: str = Field("development", env="APP_ENV")
    DEBUG: bool = Field(False, env="DEBUG")

    # Trading Configuration
    DEFAULT_SYMBOLS: List[str] = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    RISK_TOLERANCE: float = 0.05

    # API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    TWITTER_API_KEY: Optional[str] = Field(None, env="TWITTER_API_KEY")
    NEWS_API_KEY: Optional[str] = Field(None, env="NEWS_API_KEY")

    # Redis Configuration
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    # Machine Learning Configuration
    ML_MODEL_PATH: str = "./ml_models/saved_models"
    TRAINING_DATA_PATH: str = "./data/training"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create singleton settings instance
settings = Settings()
