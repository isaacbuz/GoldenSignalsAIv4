"""
API Documentation configuration for GoldenSignalsAI
"""

from typing import Dict, Any

# API Metadata
API_TITLE = "GoldenSignalsAI API"
API_VERSION = "2.0.0"
API_DESCRIPTION = """
# GoldenSignalsAI Trading Signal Platform API

## Overview
GoldenSignalsAI provides AI-powered trading signals with advanced technical analysis, 
machine learning predictions, and risk management capabilities.

## Features
- **Real-time Signal Generation** - Generate trading signals for multiple symbols
- **Technical Analysis** - 15+ technical indicators including RSI, MACD, Bollinger Bands
- **Risk Analysis** - Comprehensive risk assessment for trading signals
- **Data Quality Validation** - Ensure high-quality market data
- **Performance Tracking** - Monitor signal performance over time

## Authentication
Currently, the API is open for development. Production deployment will require API key authentication.

## Rate Limiting
- 100 requests per minute per IP address
- Burst limit: 10 concurrent requests

## Response Format
All responses are in JSON format with consistent structure:
- Successful responses include the requested data
- Error responses include error code and detailed message

## Versioning
API version is included in the URL path (e.g., `/api/v1/`)
"""

# OpenAPI Tags
TAGS_METADATA = [
    {
        "name": "signals",
        "description": "Trading signal generation and analysis endpoints",
        "externalDocs": {
            "description": "Signal Generation Documentation",
            "url": "https://goldensignalsai.com/docs/signals",
        },
    },
    {
        "name": "market",
        "description": "Market data and analysis endpoints",
        "externalDocs": {
            "description": "Market Data Documentation",
            "url": "https://goldensignalsai.com/docs/market",
        },
    },
    {
        "name": "portfolio",
        "description": "Portfolio management and simulation endpoints",
    },
    {
        "name": "risk",
        "description": "Risk analysis and management endpoints",
    },
    {
        "name": "health",
        "description": "System health and monitoring endpoints",
    },
]

# Custom OpenAPI schema
def custom_openapi_schema() -> Dict[str, Any]:
    """Generate custom OpenAPI schema with additional documentation"""
    return {
        "openapi": "3.0.2",
        "info": {
            "title": API_TITLE,
            "version": API_VERSION,
            "description": API_DESCRIPTION,
            "termsOfService": "https://goldensignalsai.com/terms",
            "contact": {
                "name": "GoldenSignalsAI Support",
                "url": "https://goldensignalsai.com/support",
                "email": "support@goldensignalsai.com"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.goldensignalsai.com",
                "description": "Production server"
            }
        ],
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key for authentication (required in production)"
                }
            },
            "responses": {
                "UnauthorizedError": {
                    "description": "API key is missing or invalid",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "detail": {
                                        "type": "string",
                                        "example": "Invalid API key"
                                    }
                                }
                            }
                        }
                    }
                },
                "RateLimitError": {
                    "description": "Rate limit exceeded",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "detail": {
                                        "type": "string",
                                        "example": "Rate limit exceeded. Try again in 60 seconds."
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


# Example responses for documentation
EXAMPLE_RESPONSES = {
    "signal": {
        "id": "AAPL_1704931200000",
        "symbol": "AAPL",
        "action": "BUY",
        "confidence": 0.75,
        "price": 185.50,
        "timestamp": "2024-01-11T00:00:00Z",
        "reason": "RSI oversold (28.5); MACD bullish crossover; Near lower Bollinger Band",
        "indicators": {
            "rsi": 28.5,
            "macd": 1.2,
            "macd_signal": 0.8,
            "sma_20": 183.25,
            "sma_50": 180.10,
            "bb_upper": 190.50,
            "bb_lower": 176.00,
            "volume_ratio": 1.25
        },
        "risk_level": "medium",
        "entry_price": 185.50,
        "stop_loss": 181.00,
        "take_profit": 195.00,
        "metadata": {
            "data_source": "yahoo",
            "data_quality_score": 0.95,
            "volume": 52341234
        },
        "quality_score": 0.85
    },
    "error": {
        "detail": "Invalid symbol provided",
        "code": "INVALID_SYMBOL",
        "timestamp": "2024-01-11T00:00:00Z"
    }
} 