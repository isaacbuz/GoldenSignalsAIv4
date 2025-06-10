"""
Enhanced Configuration Management System for GoldenSignalsAI V3

Features:
- Environment-specific configuration
- Pydantic validation with type hints
- Secure credential management
- Performance tuning parameters
- Agent-specific configurations
- Runtime configuration updates
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import os
import json
from functools import lru_cache

from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic.env_settings import SettingsSourceCallable


class Environment(str, Enum):
    """Environment enumeration"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Log level enumeration"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class CacheStrategy(str, Enum):
    """Cache strategy options"""
    MEMORY = "memory"
    REDIS = "redis"
    MULTILAYER = "multilayer"
    DISABLED = "disabled"


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    
    host: str = Field("localhost", env="DB_HOST")
    port: int = Field(5432, env="DB_PORT", ge=1, le=65535)
    database: str = Field("goldensignals", env="DB_NAME")
    username: str = Field("postgres", env="DB_USER")
    password: SecretStr = Field(..., env="DB_PASSWORD")
    pool_size: int = Field(20, env="DB_POOL_SIZE", ge=1, le=100)
    max_overflow: int = Field(30, env="DB_MAX_OVERFLOW", ge=0, le=100)
    pool_timeout: int = Field(30, env="DB_POOL_TIMEOUT", ge=1, le=300)
    pool_recycle: int = Field(3600, env="DB_POOL_RECYCLE", ge=300, le=86400)
    echo: bool = Field(False, env="DB_ECHO")
    
    @property
    def url(self) -> str:
        """Generate database URL"""
        return f"postgresql+asyncpg://{self.username}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "DB_"


class RedisConfig(BaseSettings):
    """Redis configuration"""
    
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT", ge=1, le=65535)
    database: int = Field(0, env="REDIS_DB", ge=0, le=15)
    password: Optional[SecretStr] = Field(None, env="REDIS_PASSWORD")
    max_connections: int = Field(50, env="REDIS_MAX_CONNECTIONS", ge=1, le=1000)
    socket_timeout: int = Field(5, env="REDIS_SOCKET_TIMEOUT", ge=1, le=60)
    socket_connect_timeout: int = Field(5, env="REDIS_CONNECT_TIMEOUT", ge=1, le=60)
    retry_on_timeout: bool = Field(True, env="REDIS_RETRY_ON_TIMEOUT")
    decode_responses: bool = Field(True, env="REDIS_DECODE_RESPONSES")
    
    @property
    def url(self) -> str:
        """Generate Redis URL"""
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "REDIS_"


class SecurityConfig(BaseSettings):
    """Security configuration"""
    
    secret_key: SecretStr = Field(..., env="SECRET_KEY", min_length=32)
    algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE", ge=1, le=1440)
    refresh_token_expire_days: int = Field(7, env="REFRESH_TOKEN_EXPIRE", ge=1, le=30)
    password_min_length: int = Field(8, env="PASSWORD_MIN_LENGTH", ge=6, le=128)
    max_login_attempts: int = Field(5, env="MAX_LOGIN_ATTEMPTS", ge=1, le=20)
    lockout_duration_minutes: int = Field(15, env="LOCKOUT_DURATION", ge=1, le=1440)
    
    # API Rate limiting
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE", ge=1, le=10000)
    rate_limit_burst: int = Field(100, env="RATE_LIMIT_BURST", ge=1, le=10000)
    
    class Config:
        env_prefix = "SECURITY_"


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration"""
    
    # Logging
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    log_rotation: str = Field("1 day", env="LOG_ROTATION")
    log_retention: str = Field("30 days", env="LOG_RETENTION")
    
    # Metrics
    prometheus_enabled: bool = Field(True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT", ge=1, le=65535)
    metrics_namespace: str = Field("goldensignals", env="METRICS_NAMESPACE")
    
    # Tracing
    jaeger_enabled: bool = Field(False, env="JAEGER_ENABLED")
    jaeger_endpoint: Optional[str] = Field(None, env="JAEGER_ENDPOINT")
    
    # Error tracking
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    sentry_environment: Optional[str] = Field(None, env="SENTRY_ENVIRONMENT")
    sentry_sample_rate: float = Field(0.1, env="SENTRY_SAMPLE_RATE", ge=0.0, le=1.0)
    
    class Config:
        env_prefix = "MONITORING_"


class PerformanceConfig(BaseSettings):
    """Performance optimization configuration"""
    
    # Caching
    cache_strategy: CacheStrategy = Field(CacheStrategy.MULTILAYER, env="CACHE_STRATEGY")
    cache_ttl_seconds: int = Field(300, env="CACHE_TTL", ge=1, le=86400)
    memory_cache_size: int = Field(1000, env="MEMORY_CACHE_SIZE", ge=10, le=100000)
    
    # Agent execution
    max_concurrent_agents: int = Field(50, env="MAX_CONCURRENT_AGENTS", ge=1, le=1000)
    agent_timeout_seconds: int = Field(30, env="AGENT_TIMEOUT", ge=1, le=300)
    signal_generation_timeout: int = Field(60, env="SIGNAL_TIMEOUT", ge=5, le=600)
    
    # Resource limits
    max_memory_mb: int = Field(2048, env="MAX_MEMORY_MB", ge=256, le=32768)
    max_cpu_percent: int = Field(80, env="MAX_CPU_PERCENT", ge=10, le=100)
    max_open_files: int = Field(1024, env="MAX_OPEN_FILES", ge=64, le=65536)
    
    # WebSocket configuration
    websocket_heartbeat_interval: int = Field(30, env="WS_HEARTBEAT", ge=5, le=300)
    websocket_max_connections: int = Field(1000, env="WS_MAX_CONNECTIONS", ge=1, le=10000)
    
    class Config:
        env_prefix = "PERFORMANCE_"


class TradingConfig(BaseSettings):
    """Trading-specific configuration"""
    
    # Risk management
    max_position_size_pct: float = Field(0.05, env="MAX_POSITION_SIZE", ge=0.001, le=0.5)
    max_portfolio_volatility: float = Field(0.20, env="MAX_PORTFOLIO_VOL", ge=0.01, le=1.0)
    max_correlation_threshold: float = Field(0.7, env="MAX_CORRELATION", ge=0.1, le=1.0)
    
    # Signal generation
    min_signal_confidence: float = Field(0.6, env="MIN_SIGNAL_CONFIDENCE", ge=0.0, le=1.0)
    signal_consensus_threshold: float = Field(0.7, env="CONSENSUS_THRESHOLD", ge=0.5, le=1.0)
    
    # Market data
    market_data_refresh_seconds: int = Field(60, env="MARKET_DATA_REFRESH", ge=1, le=3600)
    options_data_refresh_seconds: int = Field(300, env="OPTIONS_DATA_REFRESH", ge=60, le=3600)
    
    # Agent weights (for consensus building)
    agent_type_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "technical": 1.0,
            "fundamental": 1.2,
            "sentiment": 0.8,
            "macro": 1.1,
            "options": 1.0,
            "flow": 1.0,
            "volatility": 0.9
        },
        env="AGENT_WEIGHTS"
    )
    
    @validator('agent_type_weights', pre=True)
    def parse_agent_weights(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
    
    class Config:
        env_prefix = "TRADING_"


class AgentConfig(BaseSettings):
    """Individual agent configuration"""
    
    # Global agent settings
    enable_caching: bool = Field(True, env="AGENT_CACHING_ENABLED")
    cache_ttl: int = Field(300, env="AGENT_CACHE_TTL", ge=1, le=3600)
    retry_attempts: int = Field(3, env="AGENT_RETRY_ATTEMPTS", ge=0, le=10)
    
    # Performance monitoring
    enable_monitoring: bool = Field(True, env="AGENT_MONITORING_ENABLED")
    performance_alert_threshold: float = Field(5.0, env="PERFORMANCE_ALERT_THRESHOLD", ge=0.1, le=60.0)
    
    # Agent-specific overrides (JSON format)
    agent_overrides: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        env="AGENT_OVERRIDES"
    )
    
    @validator('agent_overrides', pre=True)
    def parse_agent_overrides(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
    
    class Config:
        env_prefix = "AGENT_"


class GoldenSignalsConfig(BaseSettings):
    """Main application configuration"""
    
    # Application metadata
    app_name: str = Field("GoldenSignalsAI", env="APP_NAME")
    version: str = Field("3.0.0", env="APP_VERSION")
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # Server configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT", ge=1, le=65535)
    workers: int = Field(4, env="WORKERS", ge=1, le=32)
    reload: bool = Field(False, env="RELOAD")
    
    # CORS configuration
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(True, env="CORS_ALLOW_CREDENTIALS")
    
    # Component configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    performance: PerformanceConfig = PerformanceConfig()
    trading: TradingConfig = TradingConfig()
    agents: AgentConfig = AgentConfig()
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        if v == Environment.PRODUCTION:
            # Additional production validations
            pass
        return v
    
    class Config:
        env_file = [".env", ".env.local"]
        env_file_encoding = 'utf-8'
        case_sensitive = False
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            # Priority: init > env > file secrets
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


class ConfigManager:
    """Configuration manager with environment-specific loading"""
    
    def __init__(self):
        self._config: Optional[GoldenSignalsConfig] = None
        self._config_file_paths = [
            Path(".env"),
            Path(".env.local"),
            Path("config") / "default.env",
        ]
    
    @property
    @lru_cache(maxsize=1)
    def config(self) -> GoldenSignalsConfig:
        """Get cached configuration instance"""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> GoldenSignalsConfig:
        """Load configuration with environment-specific overrides"""
        # Determine environment
        env = os.getenv("ENVIRONMENT", "development").lower()
        
        # Load environment-specific config file
        env_config_path = Path("config") / f"{env}.env"
        if env_config_path.exists():
            self._config_file_paths.append(env_config_path)
        
        # Create configuration instance
        config = GoldenSignalsConfig()
        
        # Apply environment-specific validations
        self._validate_environment_config(config)
        
        return config
    
    def _validate_environment_config(self, config: GoldenSignalsConfig):
        """Apply environment-specific validations"""
        if config.environment == Environment.PRODUCTION:
            # Production-specific validations
            assert not config.debug, "Debug mode must be disabled in production"
            assert config.security.secret_key.get_secret_value() != "dev-secret-key", "Production secret key required"
            assert config.database.password.get_secret_value() != "password", "Strong database password required"
            
        elif config.environment == Environment.DEVELOPMENT:
            # Development-specific settings
            config.monitoring.log_level = LogLevel.DEBUG
            config.reload = True
    
    def get_agent_config(self, agent_name: str, agent_type: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        base_config = {
            "name": agent_name,
            "agent_type": agent_type,
            "cache_enabled": self.config.agents.enable_caching,
            "cache_ttl": self.config.agents.cache_ttl,
            "retry_attempts": self.config.agents.retry_attempts,
            "monitoring_enabled": self.config.agents.enable_monitoring,
            "timeout_seconds": self.config.performance.agent_timeout_seconds,
        }
        
        # Apply agent-specific overrides
        agent_overrides = self.config.agents.agent_overrides.get(agent_name, {})
        base_config.update(agent_overrides)
        
        return base_config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration at runtime (limited scope)"""
        try:
            # Only allow updating certain non-critical settings
            allowed_updates = [
                "trading.min_signal_confidence",
                "trading.signal_consensus_threshold",
                "performance.cache_ttl_seconds",
                "monitoring.log_level"
            ]
            
            for key, value in updates.items():
                if key in allowed_updates:
                    # Apply update using dot notation
                    obj = self.config
                    parts = key.split('.')
                    
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    
                    setattr(obj, parts[-1], value)
            
            return True
            
        except Exception as e:
            print(f"Configuration update failed: {e}")
            return False
    
    def reload_config(self):
        """Reload configuration from files"""
        self._config = None
        # Clear cache
        self.config.cache_clear()
        
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for debugging"""
        return {
            "environment": self.config.environment.value,
            "debug": self.config.debug,
            "version": self.config.version,
            "host": self.config.host,
            "port": self.config.port,
            "database_host": self.config.database.host,
            "redis_host": self.config.redis.host,
            "cache_strategy": self.config.performance.cache_strategy.value,
            "log_level": self.config.monitoring.log_level.value
        }


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience function for getting config
def get_config() -> GoldenSignalsConfig:
    """Get the global configuration instance"""
    return config_manager.config 