"""
API Documentation for GoldenSignalsAI V2

This module provides comprehensive API documentation using OpenAPI/Swagger.
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI):
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="GoldenSignalsAI API",
        version="2.0.0",
        description="""
        ## Overview
        
        GoldenSignalsAI is an AI-powered financial trading platform that provides:
        
        - **Real-time Signal Generation**: 50+ specialized trading agents
        - **Risk Management**: Advanced portfolio and position risk analysis
        - **Market Analysis**: Technical, fundamental, and sentiment analysis
        - **Backtesting**: Historical performance validation
        - **ML Integration**: Transformer models and adaptive learning
        
        ## Authentication
        
        All endpoints require JWT authentication. Obtain a token via `/auth/login`.
        
        ```
        Authorization: Bearer <your-token>
        ```
        
        ## Rate Limiting
        
        - Public endpoints: 100 requests/minute
        - Authenticated: 1000 requests/minute
        - Premium: 10000 requests/minute
        
        ## WebSocket
        
        Real-time updates available at `ws://api/v1/ws`
        """,
        routes=app.routes,
        tags=[
            {
                "name": "signals",
                "description": "Trading signal generation and management"
            },
            {
                "name": "portfolio",
                "description": "Portfolio management and analytics"
            },
            {
                "name": "market",
                "description": "Market data and analysis"
            },
            {
                "name": "agents",
                "description": "AI agent management and monitoring"
            },
            {
                "name": "auth",
                "description": "Authentication and authorization"
            },
            {
                "name": "health",
                "description": "System health and monitoring"
            }
        ],
        servers=[
            {"url": "https://api.goldensignals.ai", "description": "Production"},
            {"url": "https://staging-api.goldensignals.ai", "description": "Staging"},
            {"url": "http://localhost:8000", "description": "Development"}
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add response examples
    openapi_schema["components"]["examples"] = {
        "SignalExample": {
            "value": {
                "signal_id": "sig_123",
                "symbol": "AAPL",
                "signal_type": "BUY",
                "confidence": 0.85,
                "strength": "STRONG",
                "source": "ensemble",
                "current_price": 150.25,
                "target_price": 165.00,
                "stop_loss": 145.00,
                "reasoning": "Strong technical and fundamental indicators"
            }
        },
        "ErrorExample": {
            "value": {
                "detail": "Invalid authentication credentials",
                "status_code": 401,
                "error_code": "AUTH_INVALID"
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
