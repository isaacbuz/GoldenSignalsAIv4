#!/usr/bin/env python3
"""
Enhanced Mock Backend for GoldenSignalsAI
Provides all necessary API endpoints for development
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GoldenSignalsAI Mock Backend", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")

    async def broadcast(self, message: str):
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")

manager = ConnectionManager()

# Data models
class Signal(BaseModel):
    signal_id: str
    symbol: str
    signal_type: str
    confidence: float
    source: str
    reasoning: Optional[str] = None
    created_at: str
    price: Optional[float] = None
    entry: Optional[float] = None
    target: Optional[float] = None
    stop_loss: Optional[float] = None
    agents: Optional[List[str]] = None

class LogEntry(BaseModel):
    logs: List[Dict[str, Any]]
    sessionId: str
    timestamp: str

# Mock data generators
def generate_mock_signal(symbol: str = "SPY") -> Dict[str, Any]:
    signal_types = ["BUY", "SELL", "HOLD"]
    sources = ["TechnicalAnalysisAgent", "SentimentAgent", "VolumeAgent", "MacroAgent"]

    price = random.uniform(400, 500)
    signal_type = random.choice(signal_types)
    confidence = random.uniform(65, 95)

    return {
        "signal_id": f"signal_{random.randint(1000, 9999)}",
        "symbol": symbol,
        "signal_type": signal_type,
        "confidence": round(confidence, 1),
        "source": random.choice(sources),
        "reasoning": f"AI analysis indicates {signal_type} signal based on technical patterns and market sentiment",
        "created_at": datetime.now().isoformat(),
        "price": round(price, 2),
        "entry": round(price * (1 + random.uniform(-0.02, 0.02)), 2),
        "target": round(price * (1 + random.uniform(0.03, 0.08)), 2),
        "stop_loss": round(price * (1 - random.uniform(0.02, 0.05)), 2),
        "agents": random.sample(sources, k=random.randint(1, 3))
    }

def generate_historical_data(symbol: str, timeframe: str = "1h", bars: int = 100) -> List[Dict[str, Any]]:
    """Generate mock historical price data"""
    data = []
    base_price = random.uniform(400, 500)
    now = datetime.now()

    for i in range(bars):
        timestamp = now - timedelta(hours=i)
        price = base_price + random.uniform(-5, 5)
        open_price = price + random.uniform(-2, 2)
        high = max(open_price, price) + random.uniform(0, 3)
        low = min(open_price, price) - random.uniform(0, 3)
        volume = random.randint(500000, 2000000)

        data.append({
            "time": int(timestamp.timestamp()),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(price, 2),
            "volume": volume
        })

    return list(reversed(data))  # Return in chronological order

# API Routes

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": "mock"
    }

@app.get("/api/v1/market-data/status/market")
async def get_market_status():
    """Get current market status"""
    now = datetime.now()
    # Simple market hours check (9:30 AM - 4:00 PM ET, weekdays)
    is_open = (
        now.weekday() < 5 and  # Monday = 0, Friday = 4
        9 <= now.hour < 16
    )

    return {
        "isOpen": is_open,
        "status": "open" if is_open else "closed",
        "nextOpen": "9:30 AM ET" if not is_open else None,
        "nextClose": "4:00 PM ET" if is_open else None,
        "timezone": "America/New_York",
        "timestamp": now.isoformat()
    }

@app.get("/api/v1/signals/latest")
async def get_latest_signals(limit: int = 10):
    """Get latest signals"""
    signals = [generate_mock_signal() for _ in range(limit)]
    return {
        "signals": signals,
        "total": len(signals),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/signals/{symbol}")
async def get_signals_for_symbol(symbol: str, hours_back: int = 24):
    """Get signals for a specific symbol"""
    count = random.randint(3, 8)
    signals = [generate_mock_signal(symbol) for _ in range(count)]

    return {
        "symbol": symbol,
        "signals": signals,
        "hours_back": hours_back,
        "total": len(signals),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/agents/performance")
async def get_agent_performance():
    """Get agent performance metrics"""
    agents = ["TechnicalAnalysisAgent", "SentimentAgent", "VolumeAgent", "MacroAgent", "OptionsFlowAgent"]
    performance = {}

    for agent in agents:
        performance[agent] = {
            "accuracy": round(random.uniform(75, 92), 1),
            "total_signals": random.randint(50, 200),
            "profitable_signals": random.randint(35, 150),
            "avg_return": round(random.uniform(2.5, 8.3), 2),
            "sharpe_ratio": round(random.uniform(1.2, 2.8), 2),
            "last_updated": datetime.now().isoformat()
        }

    return {
        "agents": performance,
        "overall_accuracy": round(sum(p["accuracy"] for p in performance.values()) / len(performance), 1),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/performance/overview")
async def get_performance_overview():
    """Get overall performance overview"""
    return {
        "total_signals": random.randint(500, 1500),
        "accuracy": round(random.uniform(78, 87), 1),
        "avg_return": round(random.uniform(4.2, 7.8), 2),
        "sharpe_ratio": round(random.uniform(1.5, 2.3), 2),
        "max_drawdown": round(random.uniform(-8.5, -3.2), 2),
        "win_rate": round(random.uniform(62, 78), 1),
        "profit_factor": round(random.uniform(1.4, 2.1), 2),
        "periods": {
            "1d": {"return": round(random.uniform(-2, 3), 2), "signals": random.randint(5, 15)},
            "1w": {"return": round(random.uniform(-5, 8), 2), "signals": random.randint(20, 50)},
            "1m": {"return": round(random.uniform(-8, 15), 2), "signals": random.randint(80, 200)},
            "3m": {"return": round(random.uniform(-12, 25), 2), "signals": random.randint(200, 500)},
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/market-data/historical/{symbol}")
async def get_historical_data(symbol: str, timeframe: str = "1h", bars: int = 100):
    """Get historical market data"""
    data = generate_historical_data(symbol, timeframe, bars)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": data,
        "bars": len(data),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/logs/frontend")
async def receive_frontend_logs(log_entry: LogEntry):
    """Receive and process frontend logs"""
    logger.info(f"[FRONTEND LOG] {log_entry.timestamp}: {log_entry.logs}")
    return {"status": "received", "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/search/symbols")
async def search_symbols(q: str = "", limit: int = 10):
    """Search for symbols"""
    # Mock popular symbols
    symbols = [
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "type": "ETF"},
        {"symbol": "AAPL", "name": "Apple Inc.", "type": "STOCK"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "type": "STOCK"},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "type": "STOCK"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "type": "STOCK"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "type": "STOCK"},
        {"symbol": "QQQ", "name": "Invesco QQQ Trust", "type": "ETF"},
        {"symbol": "BTC-USD", "name": "Bitcoin USD", "type": "CRYPTO"},
        {"symbol": "ETH-USD", "name": "Ethereum USD", "type": "CRYPTO"},
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "type": "STOCK"},
    ]

    if q:
        filtered = [s for s in symbols if q.upper() in s["symbol"] or q.lower() in s["name"].lower()]
    else:
        filtered = symbols

    return {
        "symbols": filtered[:limit],
        "total": len(filtered),
        "query": q
    }

# WebSocket endpoints
@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    client_id = f"signals_{random.randint(1000, 9999)}"
    await manager.connect(websocket, client_id)

    try:
        # Send initial welcome message
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "message": "Connected to signals feed",
            "client_id": client_id
        }))

        # Start sending periodic signals
        async def send_periodic_signals():
            while True:
                try:
                    await asyncio.sleep(random.uniform(10, 30))  # Random interval
                    signal = generate_mock_signal()
                    await websocket.send_text(json.dumps({
                        "type": "signal",
                        "data": signal
                    }))
                except Exception as e:
                    logger.error(f"Error sending periodic signal: {e}")
                    break

        # Start the periodic task
        task = asyncio.create_task(send_periodic_signals())

        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "subscribe":
                    symbols = message.get("symbols", ["SPY"])
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "symbols": symbols,
                        "message": f"Subscribed to {len(symbols)} symbols"
                    }))
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

        # Cleanup
        task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        manager.disconnect(client_id)

@app.websocket("/ws")
async def websocket_general(websocket: WebSocket):
    """General WebSocket endpoint - returns 403 to match expected behavior"""
    await websocket.close(code=4003, reason="Forbidden - Use /ws/signals instead")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "message": f"Endpoint {request.url.path} not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "Something went wrong"}
    )

if __name__ == "__main__":
    print("🚀 Starting GoldenSignalsAI Mock Backend")
    print("📊 Available endpoints:")
    print("  - Health: http://localhost:8000/api/v1/health")
    print("  - Market Status: http://localhost:8000/api/v1/market-data/status/market")
    print("  - Latest Signals: http://localhost:8000/api/v1/signals/latest")
    print("  - Agent Performance: http://localhost:8000/api/v1/agents/performance")
    print("  - WebSocket Signals: ws://localhost:8000/ws/signals")
    print("  - API Docs: http://localhost:8000/docs")
    print("🌐 CORS enabled for localhost:3000 and localhost:5173")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
