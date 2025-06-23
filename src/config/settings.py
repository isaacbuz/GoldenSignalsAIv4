"""
Simple settings configuration for backward compatibility
"""
import os
from typing import Optional

class Settings:
    """Simple settings class for backward compatibility"""
    
    def __init__(self):
        # Application settings
        self.app_name = os.getenv("APP_NAME", "GoldenSignalsAI")
        self.version = os.getenv("APP_VERSION", "3.0.0")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Server settings
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "4"))
        self.reload = os.getenv("RELOAD", "false").lower() == "true"
        
        # Database settings
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = int(os.getenv("DB_PORT", "5432"))
        self.db_name = os.getenv("DB_NAME", "goldensignals")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "password")
        
        # Redis settings
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD", None)
        
        # Security settings
        self.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE", "30"))
        
        # CORS settings
        cors_origins_str = os.getenv("CORS_ORIGINS", "*")
        self.cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]
        
        # Monitoring
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.prometheus_enabled = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
        self.sentry_dsn = os.getenv("SENTRY_DSN", None)
        
        # Trading settings
        self.min_signal_confidence = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.6"))
        self.max_position_size_pct = float(os.getenv("MAX_POSITION_SIZE", "0.05"))
        
        # Create composite properties
        self.database_url = f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self.redis_url = f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}" if self.redis_password else f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
        
        # Nested attributes for compatibility
        self.monitoring = type('obj', (object,), {
            'sentry_dsn': self.sentry_dsn,
            'log_level': self.log_level,
            'prometheus_enabled': self.prometheus_enabled
        })
        
        self.security = type('obj', (object,), {
            'secret_key': self.secret_key,
            'algorithm': self.algorithm,
            'access_token_expire_minutes': self.access_token_expire_minutes
        })
        
        self.database = type('obj', (object,), {
            'host': self.db_host,
            'port': self.db_port,
            'database': self.db_name,
            'username': self.db_user,
            'password': type('obj', (object,), {'get_secret_value': lambda: self.db_password})(),
            'url': self.database_url
        })
        
        self.redis = type('obj', (object,), {
            'host': self.redis_host,
            'port': self.redis_port,
            'database': self.redis_db,
            'password': self.redis_password,
            'url': self.redis_url
        })


# Global settings instance
settings = Settings()

# For backward compatibility
def get_settings() -> Settings:
    """Get settings instance"""
    return settings 