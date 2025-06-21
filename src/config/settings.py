import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Keys
    GROK_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    TWITTER_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    FINNHUB_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    BENZINGA_API_KEY: Optional[str] = None
    
    # Environment
    ENV: str = 'development'
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        
    def validate(self) -> None:
        """Validate required settings based on environment"""
        if self.ENV == 'production':
            # In production, all keys must be set
            required_keys = [
                'GROK_API_KEY',
                'ALPHA_VANTAGE_API_KEY',
                'TWITTER_API_KEY',
                'NEWS_API_KEY',
                'FINNHUB_API_KEY',
                'POLYGON_API_KEY',
                'BENZINGA_API_KEY'
            ]
        else:
            # In development, only essential keys are required
            required_keys = [
                'ALPHA_VANTAGE_API_KEY',  # For market data
                'NEWS_API_KEY'  # For news data
            ]
            
        for key in required_keys:
            if not getattr(self, key):
                raise ValueError(f"{key} environment variable is not set")

# Create settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    print(f"Warning: {str(e)}") 