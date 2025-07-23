#!/usr/bin/env python
"""Enhanced FastAPI app with market data endpoints for testing"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect
import uvicorn
import json
import asyncio
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any

app = FastAPI(title="GoldenSignalsAI Test API", version="0.2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Generate mock candlestick data
def generate_mock_candles(symbol: str, count: int = 100, interval: str = "5m") -> List[Dict[str, Any]]:
    """Generate realistic mock candlestick data"""
    candles = []
    base_price = 100.0 if symbol != "BTC-USD" else 45000.0

    # Start from count periods ago
    current_time = datetime.now()
    interval_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440
    }.get(interval, 5)

    for i in range(count):
        time_offset = (count - i) * interval_minutes
        timestamp = current_time - timedelta(minutes=time_offset)

        # Generate realistic price movement
        volatility = 0.02 if symbol != "BTC-USD" else 0.03
        change = random.uniform(-volatility, volatility)
        base_price = base_price * (1 + change)

        open_price = base_price
        close_price = base_price * (1 + random.uniform(-0.01, 0.01))
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
        volume = random.uniform(1000, 10000) if symbol != "BTC-USD" else random.uniform(100, 1000)

        candles.append({
            "time": int(timestamp.timestamp()),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 2)
        })

        base_price = close_price

    return candles

@app.get("/")
async def root():
    return {"message": "GoldenSignalsAI Test API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "GoldenSignalsAI"}

@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    return {
        "symbol": symbol,
        "price": round(random.uniform(95, 105), 2),
        "change": round(random.uniform(-5, 5), 2),
        "changePercent": round(random.uniform(-5, 5), 2),
        "volume": random.randint(1000000, 10000000),
        "high": round(random.uniform(105, 110), 2),
        "low": round(random.uniform(90, 95), 2),
        "open": round(random.uniform(98, 102), 2),
        "previousClose": round(random.uniform(98, 102), 2),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/market-data/{symbol}/history")
async def get_historical_data(
    symbol: str,
    period: str = Query("7d", description="Time period"),
    interval: str = Query("5m", description="Time interval")
):
    """Get historical market data"""
    # Map period to number of candles
    period_to_count = {
        "1d": 96,    # 96 5-minute candles in a day (assuming market hours)
        "7d": 200,   # Simplified for demo
        "1M": 500,   # Simplified for demo
        "3M": 1000,  # Simplified for demo
    }

    count = period_to_count.get(period, 100)
    candles = generate_mock_candles(symbol, count, interval)

    return {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "data": candles
    }

@app.get("/api/v1/signals/symbol/{symbol}")
async def get_signals(symbol: str):
    """Get trading signals for a symbol"""
    signals = []

    # Generate some mock signals
    signal_types = ["BUY", "SELL", "HOLD"]
    agents = ["RSI_Agent", "MACD_Agent", "Volume_Agent", "Pattern_Agent"]

    for i in range(5):
        signals.append({
            "id": f"signal_{i}",
            "symbol": symbol,
            "action": random.choice(signal_types),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "agent": random.choice(agents),
            "price": round(random.uniform(95, 105), 2),
            "timestamp": (datetime.now() - timedelta(minutes=i*10)).isoformat(),
            "reasoning": f"Technical indicator suggests {random.choice(['overbought', 'oversold', 'trend reversal', 'breakout'])} condition"
        })

    return {"signals": signals}

@app.post("/api/v1/workflow/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """Run AI workflow analysis on a symbol"""
    # Simulate AI analysis
    await asyncio.sleep(0.5)  # Simulate processing time

    return {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "market_regime": random.choice(["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE"]),
        "consensus": {
            "action": random.choice(["BUY", "SELL", "HOLD"]),
            "confidence": round(random.uniform(0.65, 0.90), 2),
            "votes": {
                "buy": random.randint(3, 7),
                "sell": random.randint(1, 4),
                "hold": random.randint(2, 5)
            }
        },
        "risk_assessment": {
            "level": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "score": round(random.uniform(0.2, 0.8), 2)
        },
        "trading_levels": {
            "entry": round(random.uniform(98, 102), 2),
            "stopLoss": round(random.uniform(92, 97), 2),
            "takeProfit": [
                round(random.uniform(103, 106), 2),
                round(random.uniform(107, 110), 2),
                round(random.uniform(111, 115), 2)
            ]
        },
        "agent_signals": [
            {
                "agent": "RSI_Agent",
                "signal": random.choice(["BUY", "SELL", "NEUTRAL"]),
                "confidence": round(random.uniform(0.6, 0.9), 2),
                "details": {"rsi": round(random.uniform(20, 80), 1)}
            },
            {
                "agent": "MACD_Agent",
                "signal": random.choice(["BUY", "SELL", "NEUTRAL"]),
                "confidence": round(random.uniform(0.6, 0.9), 2),
                "details": {"macd": round(random.uniform(-2, 2), 2)}
            },
            {
                "agent": "Pattern_Agent",
                "signal": random.choice(["BUY", "SELL", "NEUTRAL"]),
                "confidence": round(random.uniform(0.6, 0.9), 2),
                "details": {"pattern": random.choice(["Head and Shoulders", "Triangle", "Flag", "Double Top"])}
            }
        ]
    }

@app.get("/api/v1/ai/predict/{symbol}")
async def ai_predict(symbol: str):
    """Get AI prediction for a symbol"""
    current_price = 100.0

    # Generate prediction points
    prediction_points = []
    for i in range(1, 13):  # 12 points into the future
        time_offset = i * 5  # 5 minute intervals
        price_change = random.uniform(-2, 3) * (i * 0.1)  # Increasing uncertainty
        prediction_points.append({
            "time": int((datetime.now() + timedelta(minutes=time_offset)).timestamp()),
            "price": round(current_price + price_change, 2)
        })

    return {
        "symbol": symbol,
        "current_price": current_price,
        "prediction": {
            "direction": random.choice(["UP", "DOWN", "SIDEWAYS"]),
            "confidence": round(random.uniform(0.65, 0.85), 2),
            "target_price": round(current_price + random.uniform(-5, 10), 2),
            "timeframe": "1h",
            "points": prediction_points
        },
        "support_levels": [
            round(current_price - random.uniform(2, 4), 2),
            round(current_price - random.uniform(5, 8), 2)
        ],
        "resistance_levels": [
            round(current_price + random.uniform(2, 4), 2),
            round(current_price + random.uniform(5, 8), 2)
        ],
        "risk_reward_ratio": round(random.uniform(1.5, 3.5), 2),
        "confidence_bounds": {
            "upper": [p["price"] + random.uniform(1, 3) for p in prediction_points],
            "lower": [p["price"] - random.uniform(1, 3) for p in prediction_points]
        }
    }

@app.websocket("/ws/signals/{symbol}")
async def websocket_signals(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time signals"""
    await websocket.accept()
    try:
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        })

        # Send periodic updates
        while True:
            await asyncio.sleep(5)  # Send update every 5 seconds

            # Send price update
            await websocket.send_json({
                "type": "price_update",
                "symbol": symbol,
                "data": {
                    "price": round(random.uniform(98, 102), 2),
                    "volume": random.randint(1000, 5000),
                    "timestamp": datetime.now().isoformat()
                }
            })

            # Occasionally send a signal
            if random.random() > 0.7:
                await websocket.send_json({
                    "type": "signal",
                    "symbol": symbol,
                    "data": {
                        "action": random.choice(["BUY", "SELL"]),
                        "confidence": round(random.uniform(0.7, 0.9), 2),
                        "agent": random.choice(["RSI_Agent", "MACD_Agent"]),
                        "price": round(random.uniform(98, 102), 2),
                        "timestamp": datetime.now().isoformat()
                    }
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {symbol}")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.websocket("/ws/v2/signals/{symbol}")
async def websocket_v2_signals(websocket: WebSocket, symbol: str):
    """WebSocket V2 endpoint for agent signals"""
    await websocket.accept()
    try:
        await websocket.send_json({
            "type": "connected",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        })

        while True:
            await asyncio.sleep(3)

            # Send agent analysis update
            await websocket.send_json({
                "type": "agent_analysis",
                "symbol": symbol,
                "data": {
                    "agent": random.choice(["RSI_Agent", "MACD_Agent", "Volume_Agent"]),
                    "status": "completed",
                    "signal": random.choice(["BUY", "SELL", "NEUTRAL"]),
                    "confidence": round(random.uniform(0.6, 0.9), 2),
                    "timestamp": datetime.now().isoformat()
                }
            })

    except WebSocketDisconnect:
        print(f"WebSocket V2 disconnected for {symbol}")

if __name__ == "__main__":
    print("🚀 Starting enhanced test API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 Docs will be available at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")

    uvicorn.run(app, host="0.0.0.0", port=8000)
