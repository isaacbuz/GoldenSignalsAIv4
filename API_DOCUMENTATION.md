# GoldenSignalsAI V2 API Documentation

## Overview

GoldenSignalsAI provides a comprehensive REST API for accessing trading signals, market data, and portfolio management features. This document covers all available endpoints, request/response formats, and usage examples.

**Base URL**: `http://localhost:8000/api/v1`  
**API Version**: v1  
**Authentication**: Currently no authentication required (development mode)

## Table of Contents

1. [Core Endpoints](#core-endpoints)
2. [Signal Generation](#signal-generation)
3. [Market Data](#market-data)
4. [Signal Monitoring](#signal-monitoring)
5. [Pipeline Management](#pipeline-management)
6. [Backtesting](#backtesting)
7. [WebSocket API](#websocket-api)
8. [Error Handling](#error-handling)

## Core Endpoints

### Health Check

```http
GET /
```

Check if the API is operational.

**Response:**
```json
{
  "message": "GoldenSignalsAI API is running",
  "status": "operational",
  "timestamp": "2024-12-23T10:30:00Z",
  "uptime": 3600,
  "version": "2.0.0"
}
```

### Performance Metrics

```http
GET /api/v1/performance
```

Get system performance metrics including request counts, cache hits, and response times.

**Response:**
```json
{
  "requests_per_endpoint": {
    "/api/v1/signals": 1523,
    "/api/v1/market-data/SPY": 892
  },
  "cache_stats": {
    "hits": 3421,
    "misses": 876,
    "hit_rate": "79.6%"
  },
  "response_times": {
    "average": 45.2,
    "p95": 120.5,
    "p99": 250.3
  },
  "uptime_seconds": 86400
}
```

## Signal Generation

### Get All Signals

```http
GET /api/v1/signals
```

Retrieve current trading signals for monitored symbols.

**Query Parameters:**
- `symbols` (optional): Comma-separated list of symbols
- `min_confidence` (optional): Minimum confidence threshold (0-1)
- `risk_levels` (optional): Comma-separated risk levels (low,medium,high)

**Response:**
```json
{
  "signals": [
    {
      "id": "sig_123456",
      "symbol": "AAPL",
      "action": "BUY",
      "confidence": 0.85,
      "price": 185.50,
      "indicators": {
        "RSI": 45.2,
        "MACD": 0.35,
        "BB_position": 0.3
      },
      "risk_level": "medium",
      "entry_price": 185.50,
      "stop_loss": 182.00,
      "take_profit": 190.00,
      "timestamp": "2024-12-23T10:30:00Z"
    }
  ],
  "count": 15,
  "generated_at": "2024-12-23T10:30:00Z"
}
```

### Get Signal Insights

```http
GET /api/v1/signals/{symbol}/insights
```

Get detailed insights and analysis for a specific symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "current_signal": {
    "action": "BUY",
    "confidence": 0.85
  },
  "technical_analysis": {
    "trend": "bullish",
    "support_levels": [180.0, 175.0],
    "resistance_levels": [190.0, 195.0],
    "indicators": {
      "RSI": {"value": 45.2, "signal": "neutral"},
      "MACD": {"value": 0.35, "signal": "bullish"},
      "moving_averages": {
        "SMA_20": 183.50,
        "SMA_50": 181.20,
        "EMA_20": 184.10
      }
    }
  },
  "risk_metrics": {
    "volatility": 0.023,
    "beta": 1.15,
    "max_drawdown": 0.08
  }
}
```

### Generate Batch Signals

```http
POST /api/v1/signals/batch
```

Generate signals for multiple symbols in a single request.

**Request Body:**
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
  "parameters": {
    "min_confidence": 0.7,
    "include_indicators": true
  }
}
```

**Response:**
```json
{
  "signals": {
    "AAPL": { /* signal data */ },
    "GOOGL": { /* signal data */ },
    "MSFT": { /* signal data */ },
    "TSLA": { /* signal data */ }
  },
  "processing_time_ms": 245
}
```

### Get Precise Options Signals

```http
GET /api/v1/signals/precise-options
```

Get options-specific trading signals.

**Query Parameters:**
- `symbol`: Target symbol (required)
- `timeframe`: Time horizon (15m, 1h, 1d)
- `strategy`: Options strategy (covered_call, protective_put, straddle)

**Response:**
```json
{
  "symbol": "SPY",
  "options_signals": [
    {
      "contract": "SPY240126C450",
      "action": "BUY",
      "strategy": "covered_call",
      "entry_price": 2.50,
      "greeks": {
        "delta": 0.45,
        "gamma": 0.02,
        "theta": -0.08,
        "vega": 0.15
      },
      "confidence": 0.82
    }
  ]
}
```

## Market Data

### Get Real-time Quote

```http
GET /api/v1/market-data/{symbol}
```

Get real-time market data for a symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "price": 185.50,
  "change": 2.30,
  "change_percent": 1.25,
  "volume": 45238910,
  "high": 186.80,
  "low": 183.20,
  "open": 184.00,
  "previous_close": 183.20,
  "timestamp": "2024-12-23T10:30:00Z"
}
```

### Get Historical Data

```http
GET /api/v1/market-data/{symbol}/historical
```

Get historical price data.

**Query Parameters:**
- `period`: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
- `interval`: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)

**Response:**
```json
{
  "symbol": "AAPL",
  "period": "1mo",
  "interval": "1d",
  "data": [
    {
      "date": "2024-12-01",
      "open": 180.50,
      "high": 182.30,
      "low": 179.80,
      "close": 181.50,
      "volume": 50234100
    }
  ]
}
```

### Get Market Opportunities

```http
GET /api/v1/market/opportunities
```

Discover market opportunities based on technical and fundamental analysis.

**Response:**
```json
{
  "opportunities": [
    {
      "symbol": "NVDA",
      "opportunity_type": "breakout",
      "confidence": 0.88,
      "entry_price": 480.50,
      "target_price": 510.00,
      "risk_reward_ratio": 2.5,
      "indicators_aligned": 8,
      "volume_confirmation": true
    }
  ],
  "scan_timestamp": "2024-12-23T10:30:00Z"
}
```

## Signal Monitoring

### Track Signal Entry

```http
POST /api/v1/monitoring/track-entry
```

Record entry into a position based on a signal.

**Request Body:**
```json
{
  "signal_id": "sig_123456",
  "symbol": "AAPL",
  "entry_price": 185.50,
  "quantity": 100,
  "entry_time": "2024-12-23T10:30:00Z"
}
```

### Track Signal Exit

```http
POST /api/v1/monitoring/track-exit
```

Record exit from a position.

**Request Body:**
```json
{
  "signal_id": "sig_123456",
  "exit_price": 188.75,
  "exit_time": "2024-12-23T14:30:00Z",
  "reason": "target_reached"
}
```

### Get Performance Metrics

```http
GET /api/v1/monitoring/performance
```

Get comprehensive performance metrics.

**Query Parameters:**
- `timeframe`: Period for metrics (1d, 7d, 30d, all)
- `symbol` (optional): Filter by symbol

**Response:**
```json
{
  "overall": {
    "total_signals": 150,
    "win_rate": 0.68,
    "average_profit": 2.35,
    "average_loss": -1.80,
    "profit_factor": 1.85,
    "sharpe_ratio": 1.45,
    "max_drawdown": -8.5
  },
  "by_symbol": {
    "AAPL": {
      "signals": 25,
      "win_rate": 0.72,
      "avg_return": 2.8
    }
  },
  "by_strategy": {
    "momentum": {
      "signals": 80,
      "win_rate": 0.70
    }
  }
}
```

### Get Recommendations

```http
GET /api/v1/monitoring/recommendations
```

Get AI-driven improvement recommendations based on performance.

**Response:**
```json
{
  "recommendations": [
    {
      "category": "risk_management",
      "priority": "high",
      "recommendation": "Consider tightening stop losses for high volatility symbols",
      "affected_symbols": ["TSLA", "NVDA"],
      "potential_improvement": "5-8% reduction in average loss"
    },
    {
      "category": "signal_filtering",
      "priority": "medium",
      "recommendation": "Increase confidence threshold to 0.75 for better win rate",
      "current_threshold": 0.65,
      "suggested_threshold": 0.75,
      "expected_win_rate_improvement": "8%"
    }
  ]
}
```

### Get Active Signals

```http
GET /api/v1/monitoring/active-signals
```

Get currently active (open) positions.

**Response:**
```json
{
  "active_signals": [
    {
      "signal_id": "sig_789012",
      "symbol": "GOOGL",
      "entry_price": 140.25,
      "current_price": 142.50,
      "unrealized_pnl": 225.00,
      "unrealized_pnl_percent": 1.60,
      "duration_hours": 3.5,
      "stop_loss": 137.50,
      "take_profit": 145.00
    }
  ],
  "count": 5,
  "total_unrealized_pnl": 850.00
}
```

## Pipeline Management

### Get Pipeline Statistics

```http
GET /api/v1/pipeline/stats
```

Get statistics about the signal filtering pipeline.

**Response:**
```json
{
  "filter_stats": {
    "ConfidenceFilter": {
      "processed": 1000,
      "passed": 750,
      "rejected": 250,
      "pass_rate": 0.75
    },
    "QualityScoreFilter": {
      "processed": 750,
      "passed": 650,
      "rejected": 100,
      "pass_rate": 0.87
    }
  },
  "total_processed": 1000,
  "total_passed": 450,
  "overall_pass_rate": 0.45
}
```

### Configure Pipeline

```http
POST /api/v1/pipeline/configure
```

Configure signal filtering pipeline parameters.

**Request Body:**
```json
{
  "filters": {
    "confidence": {
      "enabled": true,
      "min_confidence": 0.75
    },
    "quality": {
      "enabled": true,
      "min_quality": 0.80
    },
    "risk": {
      "enabled": true,
      "allowed_levels": ["low", "medium"]
    }
  }
}
```

### Get Signal Quality Report

```http
GET /api/v1/signals/quality-report
```

Get detailed quality analysis of generated signals.

**Response:**
```json
{
  "quality_metrics": {
    "average_confidence": 0.78,
    "high_quality_ratio": 0.65,
    "data_quality_score": 0.92
  },
  "quality_breakdown": {
    "excellent": 45,
    "good": 120,
    "fair": 80,
    "poor": 25
  },
  "recommendations": [
    "Consider increasing data refresh rate for volatile symbols",
    "Review indicator weightings for MACD signals"
  ]
}
```

## Backtesting

### Run Backtest

```http
POST /api/v1/backtest/run
```

Run ML-enhanced backtest with signal quality tracking.

**Request Body:**
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-01",
  "initial_capital": 100000,
  "parameters": {
    "position_size": 0.1,
    "max_positions": 5,
    "use_ml_signals": true,
    "confidence_threshold": 0.75
  }
}
```

**Response:**
```json
{
  "results": {
    "total_return": 0.185,
    "sharpe_ratio": 1.52,
    "max_drawdown": -0.085,
    "win_rate": 0.68,
    "total_trades": 245,
    "profitable_trades": 167
  },
  "by_symbol": {
    "AAPL": {
      "return": 0.22,
      "trades": 85,
      "win_rate": 0.71
    }
  },
  "ml_performance": {
    "signal_accuracy": 0.72,
    "feature_importance": {
      "RSI": 0.18,
      "MACD": 0.15,
      "volume_ratio": 0.12
    }
  }
}
```

### Get Backtest Recommendations

```http
GET /api/v1/backtest/recommendations
```

Get recommendations based on backtest results.

**Response:**
```json
{
  "parameter_optimization": {
    "suggested_confidence": 0.78,
    "suggested_position_size": 0.08,
    "expected_improvement": "12% higher Sharpe ratio"
  },
  "strategy_recommendations": [
    "Add volume confirmation for breakout signals",
    "Consider shorter holding periods for high volatility symbols"
  ]
}
```

## WebSocket API

### Real-time Signal Updates

```websocket
ws://localhost:8000/ws
```

Connect to receive real-time signal updates.

**Connection Message:**
```json
{
  "type": "subscribe",
  "channels": ["signals", "market_data"],
  "symbols": ["AAPL", "GOOGL", "SPY"]
}
```

**Signal Update:**
```json
{
  "type": "signal_update",
  "data": {
    "symbol": "AAPL",
    "action": "BUY",
    "confidence": 0.85,
    "timestamp": "2024-12-23T10:30:00Z"
  }
}
```

**Market Data Update:**
```json
{
  "type": "market_update",
  "data": {
    "symbol": "AAPL",
    "price": 185.75,
    "change": 0.25,
    "volume": 125000
  }
}
```

## Error Handling

All endpoints follow a consistent error response format:

```json
{
  "error": "Error message",
  "detail": "Detailed error description",
  "code": "ERROR_CODE",
  "timestamp": "2024-12-23T10:30:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|------------|-------------|
| `INVALID_SYMBOL` | 400 | Invalid or unsupported symbol |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `DATA_UNAVAILABLE` | 503 | Market data temporarily unavailable |
| `INVALID_PARAMETERS` | 422 | Invalid request parameters |
| `INTERNAL_ERROR` | 500 | Internal server error |

### Rate Limiting

API requests are rate limited to:
- 60 requests per minute per IP
- 1000 requests per hour per IP

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Examples

### Python Example

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000/api/v1"

# Get signals
response = requests.get(f"{BASE_URL}/signals")
signals = response.json()

for signal in signals["signals"]:
    print(f"{signal['symbol']}: {signal['action']} (confidence: {signal['confidence']})")

# Track signal entry
entry_data = {
    "signal_id": signals["signals"][0]["id"],
    "symbol": signals["signals"][0]["symbol"],
    "entry_price": signals["signals"][0]["price"],
    "quantity": 100,
    "entry_time": "2024-12-23T10:30:00Z"
}

response = requests.post(f"{BASE_URL}/monitoring/track-entry", json=entry_data)
print("Entry tracked:", response.json())
```

### JavaScript Example

```javascript
// Fetch signals
fetch('http://localhost:8000/api/v1/signals')
  .then(response => response.json())
  .then(data => {
    data.signals.forEach(signal => {
      console.log(`${signal.symbol}: ${signal.action} (${signal.confidence})`);
    });
  });

// WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['signals'],
    symbols: ['AAPL', 'GOOGL']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

## Changelog

### Version 2.0.0 (December 2024)
- Added comprehensive signal generation engine
- Implemented 7-stage signal filtering pipeline
- Added signal monitoring and performance tracking
- Enhanced backtesting with ML integration
- Added WebSocket support for real-time updates
- Improved error handling and rate limiting 