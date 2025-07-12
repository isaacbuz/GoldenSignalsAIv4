"""
CORS configuration for production security.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
from urllib.parse import urlparse

def get_allowed_origins() -> List[str]:
    """Get allowed origins from environment or defaults."""
    # Production domains
    production_origins = [
        "https://goldensignals.ai",
        "https://www.goldensignals.ai",
        "https://app.goldensignals.ai",
        "https://api.goldensignals.ai"
    ]
    
    # Development origins
    development_origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]
    
    # Get environment
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        # In production, only allow production origins
        allowed = production_origins
    elif environment == "staging":
        # In staging, allow staging and development
        allowed = production_origins + development_origins + [
            "https://staging.goldensignals.ai",
            "https://staging-app.goldensignals.ai"
        ]
    else:
        # In development, allow all configured origins
        allowed = production_origins + development_origins
    
    # Add any custom origins from environment
    custom_origins = os.getenv("CORS_ORIGINS", "").split(",")
    custom_origins = [origin.strip() for origin in custom_origins if origin.strip()]
    allowed.extend(custom_origins)
    
    return list(set(allowed))  # Remove duplicates

def configure_cors(app: FastAPI) -> None:
    """Configure CORS for the FastAPI application."""
    allowed_origins = get_allowed_origins()
    
    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
            "X-Correlation-ID",
            "X-Client-Version"
        ],
        expose_headers=[
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Request-ID",
            "X-Process-Time"
        ],
        max_age=86400  # 24 hours
    )

def is_origin_allowed(origin: str, allowed_origins: List[str]) -> bool:
    """Check if an origin is allowed."""
    if not origin:
        return False
    
    # Parse the origin
    parsed = urlparse(origin)
    origin_host = parsed.netloc.lower()
    origin_scheme = parsed.scheme.lower()
    
    for allowed in allowed_origins:
        allowed_parsed = urlparse(allowed)
        allowed_host = allowed_parsed.netloc.lower()
        allowed_scheme = allowed_parsed.scheme.lower()
        
        # Check exact match
        if origin == allowed:
            return True
        
        # Check wildcard subdomain match (*.goldensignals.ai)
        if allowed_host.startswith("*."):
            domain = allowed_host[2:]  # Remove *.
            if origin_host.endswith(domain) and origin_scheme == allowed_scheme:
                return True
    
    return False

class CORSConfig:
    """CORS configuration class for advanced settings."""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.allowed_origins = get_allowed_origins()
        self.allow_credentials = True
        self.max_age = 86400
    
    def add_origin(self, origin: str) -> None:
        """Add an allowed origin."""
        if origin not in self.allowed_origins:
            self.allowed_origins.append(origin)
    
    def remove_origin(self, origin: str) -> None:
        """Remove an allowed origin."""
        if origin in self.allowed_origins:
            self.allowed_origins.remove(origin)
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    def get_policy(self) -> dict:
        """Get CORS policy as dictionary."""
        return {
            "allowed_origins": self.allowed_origins,
            "allow_credentials": self.allow_credentials,
            "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "allowed_headers": [
                "Authorization",
                "Content-Type",
                "X-API-Key",
                "X-Request-ID"
            ],
            "max_age": self.max_age,
            "environment": self.environment
        }

# Global CORS configuration instance
cors_config = CORSConfig() 