from typing import List, Optional, Union, Any
from pydantic import Field, AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_NAME: str = "GoldenSignalsAI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "your-ultra-secure-secret-key-change-this-in-production"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://goldensignals:your_secure_password@localhost:5432/goldensignals"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "goldensignals"
    DB_USER: str = "goldensignals"
    DB_PASSWORD: str = "your_secure_password"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None

    # CORS
    CORS_ORIGINS: Union[str, List[AnyHttpUrl]] = "http://localhost:3000,http://localhost:3001"

    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        return v

    # JWT
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    FINNHUB_API_KEY: Optional[str] = None

    # Feature Flags
    ENABLE_LIVE_DATA: bool = True
    ENABLE_AI_CHAT: bool = True
    ENABLE_OPTIONS_TRADING: bool = True
    ENABLE_BACKTESTING: bool = True
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = False
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Trading
    DEFAULT_SYMBOLS: str = "AAPL,GOOGL,MSFT,AMZN,SPY,TSLA,NVDA,META"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore'
    )


# Create singleton settings instance
settings = Settings()
