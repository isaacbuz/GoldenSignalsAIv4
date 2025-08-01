#!/usr/bin/env python3
"""
Simple Backend for Chart Testing
Provides the market data endpoints needed by the frontend
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chart Testing Backend", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

def generate_historical_data(symbol: str, period: str = "5d", interval: str = "15m") -> List[Dict]:
    """Generate realistic historical candlestick data"""
    # Base prices for different symbols
    base_prices = {
        "AAPL": 195.0,
        "GOOGL": 175.0,
        "MSFT": 430.0,
        "AMZN": 185.0,
        "TSLA": 250.0,
    }

    base_price = base_prices.get(symbol, 100.0)

    # Calculate number of bars based on period and interval
    if period == "1d" and interval == "5m":
        num_bars = 78  # ~6.5 hours of trading
    elif period == "5d" and interval == "15m":
        num_bars = 130  # 5 days * 26 bars per day
    else:
        num_bars = 100

    data = []
    current_time = datetime.now()

    for i in range(num_bars):
        # Time calculation
        time_offset = timedelta(minutes=15 * (num_bars - i))
        bar_time = current_time - time_offset

        # Price simulation with trend
        trend = 0.0001 * i  # Slight upward trend
        volatility = base_price * 0.002  # 0.2% volatility

        open_price = base_price + trend * base_price + random.uniform(-volatility, volatility)
        close_price = open_price + random.uniform(-volatility, volatility)
        high_price = max(open_price, close_price) + random.uniform(0, volatility)
        low_price = min(open_price, close_price) - random.uniform(0, volatility)

        volume = random.randint(1000000, 5000000)

        data.append({
            "time": int(bar_time.timestamp()),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume
        })

        # Update base price for next bar
        base_price = close_price

    return data

@app.get("/")
async def root():
    return {"message": "Chart Testing Backend is running"}

@app.get("/api/v1/market-data/{symbol}/historical")
async def get_historical_data(symbol: str, period: str = "5d", interval: str = "15m"):
    """Get historical market data for a symbol"""
    data = generate_historical_data(symbol.upper(), period, interval)
    return {"data": data, "symbol": symbol.upper(), "period": period, "interval": interval}

@app.get("/api/v1/market-data/historical/{symbol}")
async def get_historical_data_alt(symbol: str, timeframe: str = "1h", bars: int = 100):
    """Alternative endpoint for historical data"""
    # Convert timeframe to period/interval
    if timeframe == "1h":
        period = "5d"
        interval = "1h"
    else:
        period = "5d"
        interval = "15m"

    data = generate_historical_data(symbol.upper(), period, interval)
    # Limit to requested number of bars
    data = data[-bars:] if len(data) > bars else data
    return data

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "WebSocket connected successfully"
        })

        # Simulate real-time price updates
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        while True:
            await asyncio.sleep(2)  # Update every 2 seconds

            # Generate random price update
            symbol = random.choice(symbols)
            price = random.uniform(100, 500)
            change = random.uniform(-5, 5)

            update = {
                "type": "price_update",
                "symbol": symbol,
                "price": round(price, 2),
                "change": round(change, 2),
                "changePercent": round(change / price * 100, 2),
                "timestamp": datetime.now().isoformat()
            }

            await manager.broadcast(update)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    print("🚀 Starting Chart Testing Backend on http://localhost:8000")
    print("📊 Historical Data: http://localhost:8000/api/v1/market-data/{symbol}/historical")
    print("🔄 WebSocket: ws://localhost:8000/ws")
    print("📚 API Docs: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
