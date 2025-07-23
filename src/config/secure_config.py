"""
Secure Configuration Management
Handles environment variables with validation and secure defaults
"""

import logging
import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing"""

    pass


class SecureConfig:
    """Secure configuration management with validation"""

    def __init__(self):
        self._validate_environment()

    def _validate_environment(self):
        """Validate that .env file is not in git"""
        git_dir = Path(".git")
        if git_dir.exists():
            # Check if .env is tracked
            try:
                result = os.popen("git ls-files .env").read().strip()
                if result:
                    raise ConfigurationError(
                        "CRITICAL: .env file is tracked in git! "
                        "Run scripts/security_cleanup.sh immediately!"
                    )
            except Exception as e:
                logger.warning(f"Could not check git status: {e}")

    @staticmethod
    def get_required(key: str) -> str:
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ConfigurationError(
                f"Missing required environment variable: {key}. " f"Please check your .env file."
            )
        return value

    @staticmethod
    def get_optional(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get optional environment variable with default"""
        return os.getenv(key, default)

    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """Get integer environment variable"""
        value = os.getenv(key)
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            raise ConfigurationError(f"Invalid integer value for {key}: {value}")

    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """Get boolean environment variable"""
        value = os.getenv(key, "").lower()
        if not value:
            return default
        return value in ("true", "1", "yes", "on")

    @staticmethod
    def get_list(key: str, separator: str = ",") -> list:
        """Get list from comma-separated environment variable"""
        value = os.getenv(key, "")
        if not value:
            return []
        return [item.strip() for item in value.split(separator)]

    @staticmethod
    def generate_secret_key() -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(32)

    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate all API keys are present"""
        required_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "XAI_API_KEY",
        ]

        optional_keys = [
            "TWELVEDATA_API_KEY",
            "FINNHUB_API_KEY",
            "ALPHA_VANTAGE_API_KEY",
            "POLYGON_API_KEY",
            "FMP_API_KEY",
        ]

        status = {}

        # Check required keys
        for key in required_keys:
            try:
                value = cls.get_required(key)
                # Basic validation - check if it looks like a real key
                if value.startswith("sk-") or len(value) > 20:
                    status[key] = True
                else:
                    status[key] = False
                    logger.warning(f"{key} appears to be a placeholder")
            except ConfigurationError:
                status[key] = False
                logger.error(f"Missing required API key: {key}")

        # Check optional keys
        for key in optional_keys:
            value = cls.get_optional(key)
            status[key] = bool(value and len(value) > 10)

        return status

    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL with validation"""
        # Try DATABASE_URL first
        db_url = cls.get_optional("DATABASE_URL")
        if db_url:
            return db_url

        # Build from components
        host = cls.get_optional("DATABASE_HOST", "localhost")
        port = cls.get_int("DATABASE_PORT", 5432)
        name = cls.get_optional("DATABASE_NAME", "goldensignalsai")
        user = cls.get_optional("DATABASE_USER", "goldensignalsai")
        password = cls.get_optional("DATABASE_PASSWORD", "")

        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{name}"
        else:
            return f"postgresql://{user}@{host}:{port}/{name}"

    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis URL with validation"""
        host = cls.get_optional("REDIS_HOST", "localhost")
        port = cls.get_int("REDIS_PORT", 6379)
        password = cls.get_optional("REDIS_PASSWORD")
        db = cls.get_int("REDIS_DB", 0)

        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"


# Singleton instance
config = SecureConfig()


# Application settings
APP_NAME = config.get_optional("APP_NAME", "GoldenSignalsAI")
VERSION = config.get_optional("VERSION", "1.0.0")
ENVIRONMENT = config.get_optional("ENVIRONMENT", "development")
DEBUG = config.get_bool("DEBUG", False)
SECRET_KEY = config.get_optional("SECRET_KEY", config.generate_secret_key())

# Database settings
DATABASE_URL = config.get_database_url()

# Redis settings
REDIS_URL = config.get_redis_url()

# API Keys
OPENAI_API_KEY = config.get_optional("OPENAI_API_KEY")
ANTHROPIC_API_KEY = config.get_optional("ANTHROPIC_API_KEY")
XAI_API_KEY = config.get_optional("XAI_API_KEY")

# Data provider keys
TWELVEDATA_API_KEY = config.get_optional("TWELVEDATA_API_KEY")
FINNHUB_API_KEY = config.get_optional("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = config.get_optional("ALPHA_VANTAGE_API_KEY")
POLYGON_API_KEY = config.get_optional("POLYGON_API_KEY")
FMP_API_KEY = config.get_optional("FMP_API_KEY")

# Security settings
JWT_SECRET_KEY = config.get_optional("JWT_SECRET_KEY", config.generate_secret_key())
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = config.get_int("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30)
JWT_REFRESH_TOKEN_EXPIRE_DAYS = config.get_int("JWT_REFRESH_TOKEN_EXPIRE_DAYS", 30)

# CORS settings
CORS_ORIGINS = config.get_list("CORS_ORIGINS")

# Monitoring
SENTRY_DSN = config.get_optional("SENTRY_DSN")

# LangSmith
LANGCHAIN_API_KEY = config.get_optional("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = config.get_bool("LANGCHAIN_TRACING_V2", False)
LANGCHAIN_PROJECT = config.get_optional("LANGCHAIN_PROJECT")

# MCP Settings
MCP_GATEWAY_URL = config.get_optional("MCP_GATEWAY_URL", "http://localhost:8001")
MCP_API_KEY = config.get_optional("MCP_API_KEY", config.generate_secret_key())


def validate_configuration():
    """Validate all configuration on startup"""
    logger.info("Validating configuration...")

    # Check API keys
    api_status = config.validate_api_keys()
    valid_keys = sum(1 for v in api_status.values() if v)
    total_keys = len(api_status)

    logger.info(f"API Keys: {valid_keys}/{total_keys} configured")
    for key, status in api_status.items():
        if not status:
            logger.warning(f"Missing or invalid: {key}")

    # Validate critical settings
    if SECRET_KEY == config.generate_secret_key():
        logger.warning("Using generated SECRET_KEY - set a permanent one in production")

    if JWT_SECRET_KEY == config.generate_secret_key():
        logger.warning("Using generated JWT_SECRET_KEY - set a permanent one in production")

    if ENVIRONMENT == "production" and DEBUG:
        raise ConfigurationError("DEBUG must be False in production!")

    logger.info("Configuration validation complete")

    return api_status


# Export all configuration variables
__all__ = [
    "config",
    "validate_configuration",
    "APP_NAME",
    "VERSION",
    "ENVIRONMENT",
    "DEBUG",
    "SECRET_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "TWELVEDATA_API_KEY",
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "POLYGON_API_KEY",
    "FMP_API_KEY",
    "JWT_SECRET_KEY",
    "JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
    "JWT_REFRESH_TOKEN_EXPIRE_DAYS",
    "CORS_ORIGINS",
    "SENTRY_DSN",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT",
    "MCP_GATEWAY_URL",
    "MCP_API_KEY",
]
