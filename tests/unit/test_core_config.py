"""
Unit tests for core configuration
"""
import pytest
import os
from unittest.mock import patch
from src.config.settings import Settings, get_settings


class TestCoreConfig:
    """Test cases for core configuration"""
    
    def test_settings_initialization(self):
        """Test settings initialization with defaults"""
        settings = Settings()
        
        # Check defaults
        assert settings.debug == False
        assert settings.app_name == "GoldenSignalsAI"
        assert settings.version == "3.0.0"
        assert settings.environment == "development"
    
    @patch.dict(os.environ, {"DEBUG": "true", "APP_NAME": "TestApp"})
    def test_settings_from_env(self):
        """Test settings loading from environment variables"""
        settings = Settings()
        
        assert settings.debug == True
        assert settings.app_name == "TestApp"
    
    def test_database_configuration(self):
        """Test database configuration"""
        settings = Settings()
        
        # Should have database settings
        assert settings.db_host == "localhost"
        assert settings.db_port == 5432
        assert settings.db_name == "goldensignals"
        assert settings.db_user == "postgres"
        assert settings.database_url is not None
        assert "postgresql" in settings.database_url
    
    def test_redis_configuration(self):
        """Test Redis configuration"""
        settings = Settings()
        
        # Should have Redis settings
        assert settings.redis_host == "localhost"
        assert settings.redis_port == 6379
        assert settings.redis_url is not None
        assert "redis://" in settings.redis_url
    
    def test_security_configuration(self):
        """Test security configuration"""
        settings = Settings()
        
        # Security settings
        assert settings.secret_key is not None
        assert settings.algorithm == "HS256"
        assert settings.access_token_expire_minutes == 30
    
    def test_cors_configuration(self):
        """Test CORS configuration"""
        settings = Settings()
        
        # CORS should be configured
        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) > 0
    
    def test_nested_attributes(self):
        """Test nested attribute compatibility"""
        settings = Settings()
        
        # Test monitoring nested object
        assert hasattr(settings, 'monitoring')
        assert settings.monitoring.sentry_dsn is None
        assert settings.monitoring.log_level == "INFO"
        
        # Test security nested object
        assert hasattr(settings, 'security')
        assert settings.security.secret_key is not None
        
        # Test database nested object
        assert hasattr(settings, 'database')
        assert settings.database.host == "localhost"
        assert callable(settings.database.password.get_secret_value)
        
        # Test redis nested object
        assert hasattr(settings, 'redis')
        assert settings.redis.host == "localhost"
    
    def test_singleton_pattern(self):
        """Test that get_settings returns singleton"""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should be the same instance
        assert settings1 is settings2
    
    @patch.dict(os.environ, {"ENVIRONMENT": "production", "DEBUG": "false"})
    def test_environment_specific_settings(self):
        """Test environment-specific settings"""
        settings = Settings()
        
        assert settings.environment == "production"
        assert settings.debug == False
    
    def test_trading_configuration(self):
        """Test trading configuration"""
        settings = Settings()
        
        # Trading settings
        assert settings.min_signal_confidence == 0.6
        assert settings.max_position_size_pct == 0.05
        assert 0 <= settings.min_signal_confidence <= 1
        assert 0 < settings.max_position_size_pct < 1 