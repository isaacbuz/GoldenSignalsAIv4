"""
Configuration management for GoldenSignalsAI V3
"""

from functools import lru_cache
from typing import Dict, List, Optional, Union

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    
    # Database type - auto-detects for development
    db_type: str = Field(default="auto", env="DB_TYPE")  # "auto", "postgresql", "sqlite"
    
    # PostgreSQL configuration
    url: str = Field(default="postgresql+asyncpg://localhost/goldensignals", env="DATABASE_URL")
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    echo: bool = Field(default=False, env="DB_ECHO")
    
    # SQLite configuration (fallback for development)
    sqlite_path: str = Field(default="./goldensignals.db", env="SQLITE_PATH")
    
    @property
    def effective_url(self) -> str:
        """Get the effective database URL based on environment and availability"""
        if self.db_type == "sqlite":
            return f"sqlite+aiosqlite:///{self.sqlite_path}"
        elif self.db_type == "postgresql":
            return self.url
        else:  # auto mode
            # In auto mode, we'll try PostgreSQL first, then fallback to SQLite
            # This will be handled in the database manager
            return self.url
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database"""
        return self.db_type == "sqlite" or (self.db_type == "auto" and "sqlite" in self.effective_url)


class RedisConfig(BaseSettings):
    """Redis configuration for caching and real-time data"""
    
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")


class AIConfig(BaseSettings):
    """AI and ML model configuration"""
    
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    default_llm: str = Field(default="gpt-4", env="DEFAULT_LLM")
    
    # Model settings
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    
    # Agent settings
    max_concurrent_agents: int = Field(default=10, env="MAX_CONCURRENT_AGENTS")
    agent_timeout: int = Field(default=30, env="AGENT_TIMEOUT")


class TradingConfig(BaseSettings):
    """Trading and market data configuration"""
    
    # Data providers
    alpha_vantage_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_KEY")
    polygon_key: Optional[str] = Field(default=None, env="POLYGON_KEY")
    alpaca_key: Optional[str] = Field(default=None, env="ALPACA_KEY")
    alpaca_secret: Optional[str] = Field(default=None, env="ALPACA_SECRET")
    
    # Risk management
    max_portfolio_risk: float = Field(default=0.05, env="MAX_PORTFOLIO_RISK")
    max_position_size: float = Field(default=0.10, env="MAX_POSITION_SIZE")
    stop_loss_percentage: float = Field(default=2.0, env="STOP_LOSS_PERCENTAGE")
    take_profit_percentage: float = Field(default=6.0, env="TAKE_PROFIT_PERCENTAGE")
    
    # Strategy settings
    lookback_period: int = Field(default=60, env="LOOKBACK_PERIOD")
    rebalance_frequency: str = Field(default="daily", env="REBALANCE_FREQUENCY")
    
    # Real-time data
    update_interval: int = Field(default=1, env="UPDATE_INTERVAL")  # seconds
    max_websocket_connections: int = Field(default=100, env="MAX_WS_CONNECTIONS")


class SecurityConfig(BaseSettings):
    """Security and authentication configuration"""
    
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration"""
    
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Performance monitoring
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")


class Settings(BaseSettings):
    """Main application settings"""
    
    # App metadata
    app_name: str = Field(default="GoldenSignalsAI V3", env="APP_NAME")
    version: str = Field(default="3.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        env="CORS_ORIGINS"
    )
    
    # Component configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    ai: AIConfig = AIConfig()
    trading: TradingConfig = TradingConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables
        
    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings() 