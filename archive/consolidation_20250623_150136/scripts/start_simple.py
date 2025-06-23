#!/usr/bin/env python3
"""
Simple Backend Server - Minimal working version
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import yfinance as yf
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active connections
active_connections: List[WebSocket] = []

# Simple cache for market data
market_data_cache = {}
cache_ttl = 300  # 5 minutes

def now_utc():
    """Get current UTC time with timezone info"""
    return datetime.utcnow().replace(tzinfo=None)

async def fetch_live_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch live data from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get current price
        current_price = info.get('regularMarketPrice') or info.get('ask') or info.get('bid') or 0
        
        return {
            "symbol": symbol,
            "price": current_price,
            "change": info.get('regularMarketChange', 0),
            "changePercent": info.get('regularMarketChangePercent', 0),
            "volume": info.get('regularMarketVolume', 0),
            "high": info.get('regularMarketDayHigh', current_price),
            "low": info.get('regularMarketDayLow', current_price),
            "open": info.get('regularMarketOpen', current_price),
            "previousClose": info.get('regularMarketPreviousClose', current_price),
            "timestamp": now_utc().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def generate_mock_signal():
    """Generate a mock trading signal"""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "SPY"]
    patterns = ["Bull Flag", "Ascending Triangle", "Double Bottom", "Cup and Handle"]
    
    symbol = random.choice(symbols)
    pattern = random.choice(patterns)
    
    return {
        "id": f"{symbol}_{int(now_utc().timestamp())}_{random.randint(1, 100)}",
        "symbol": symbol,
        "pattern": pattern,
        "confidence": round(random.uniform(70, 95), 1),
        "entry": round(random.uniform(100, 500), 2),
        "stopLoss": round(random.uniform(95, 495), 2),
        "takeProfit": round(random.uniform(105, 505), 2),
        "timestamp": now_utc().isoformat(),
        "type": random.choice(["BUY", "SELL"]),
        "timeframe": random.choice(["5m", "15m", "1h", "4h", "1d"]),
        "risk": random.choice(["LOW", "MEDIUM", "HIGH"]),
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Simple Backend Server")
    yield
    # Shutdown
    logger.info("Shutting down Simple Backend Server")

app = FastAPI(
    title="GoldenSignalsAI Simple Backend",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "GoldenSignalsAI Simple Backend", "status": "running"}

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": now_utc().isoformat(),
        "services": {
            "websocket_connections": len(active_connections),
        }
    }

@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    # Check cache first
    cache_key = f"quote_{symbol}"
    if cache_key in market_data_cache:
        cached_data, cached_time = market_data_cache[cache_key]
        if (now_utc() - cached_time).seconds < cache_ttl:
            return cached_data
    
    # Fetch live data
    data = await fetch_live_data(symbol)
    if data:
        market_data_cache[cache_key] = (data, now_utc())
        return data
    
    # Fallback to mock data
    return {
        "symbol": symbol,
        "price": round(random.uniform(100, 500), 2),
        "change": round(random.uniform(-5, 5), 2),
        "changePercent": round(random.uniform(-2, 2), 2),
        "volume": random.randint(1000000, 50000000),
        "timestamp": now_utc().isoformat(),
    }

@app.get("/api/v1/market-data/{symbol}/historical")
async def get_historical_data(symbol: str, period: str = "1d", interval: str = "5m"):
    """Get historical market data"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if not hist.empty:
            data = []
            for index, row in hist.iterrows():
                data.append({
                    "time": int(index.timestamp()),
                    "open": round(row['Open'], 2),
                    "high": round(row['High'], 2),
                    "low": round(row['Low'], 2),
                    "close": round(row['Close'], 2),
                    "volume": int(row['Volume']),
                })
            return {"data": data, "symbol": symbol, "period": period, "interval": interval}
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
    
    # Return mock data as fallback
    data_points = 100
    data = []
    base_price = 150.0
    current_time = now_utc()
    
    for i in range(data_points, 0, -1):
        time = current_time - timedelta(minutes=i * 5)
        open_price = base_price + random.uniform(-2, 2)
        close_price = open_price + random.uniform(-1, 1)
        
        data.append({
            "time": int(time.timestamp()),
            "open": round(open_price, 2),
            "high": round(max(open_price, close_price) + random.uniform(0, 0.5), 2),
            "low": round(min(open_price, close_price) - random.uniform(0, 0.5), 2),
            "close": round(close_price, 2),
            "volume": random.randint(100000, 1000000),
        })
        base_price = close_price
    
    return {"data": data, "symbol": symbol, "period": period, "interval": interval}

@app.get("/api/v1/signals")
async def get_signals():
    """Get latest trading signals"""
    signals = [generate_mock_signal() for _ in range(10)]
    return {"signals": signals, "count": len(signals)}

@app.get("/api/v1/signals/{symbol}")
async def get_symbol_signals(symbol: str):
    """Get signals for a specific symbol"""
    signals = [generate_mock_signal() for _ in range(5)]
    for signal in signals:
        signal["symbol"] = symbol
    return {"signals": signals, "symbol": symbol}

@app.get("/api/v1/signals/{signal_id}/insights")
async def get_signal_insights(signal_id: str):
    """Get AI insights for a specific signal"""
    symbol = signal_id.split('_')[0] if '_' in signal_id else 'SPY'
    
    return {
        "signalId": signal_id,
        "insights": {
            "sentiment": {
                "score": round(random.uniform(0.6, 0.9), 2),
                "label": random.choice(["Bullish", "Very Bullish", "Neutral"]),
                "sources": ["Twitter", "Reddit", "News"],
            },
            "technicalAnalysis": {
                "rsi": round(random.uniform(30, 70), 1),
                "macd": random.choice(["Bullish Crossover", "Bearish Divergence", "Neutral"]),
                "supportLevels": [round(random.uniform(90, 98), 2) for _ in range(3)],
                "resistanceLevels": [round(random.uniform(102, 110), 2) for _ in range(3)],
            },
            "aiPrediction": {
                "targetPrice": round(random.uniform(105, 120), 2),
                "confidence": round(random.uniform(70, 90), 1),
                "timeframe": "7 days",
                "reasoning": "Strong momentum with breakout pattern confirmed by volume",
            },
            "riskAnalysis": {
                "volatility": random.choice(["Low", "Medium", "High"]),
                "maxDrawdown": round(random.uniform(2, 8), 1),
                "sharpeRatio": round(random.uniform(1.0, 2.5), 2),
            },
        },
    }

@app.get("/api/v1/market/opportunities")
async def get_market_opportunities():
    """Get current market opportunities"""
    opportunities = []
    symbols_data = [
        ("NVDA", "NVIDIA Corporation", "Technology"),
        ("TSLA", "Tesla Inc", "Automotive"),
        ("META", "Meta Platforms Inc", "Technology"),
        ("SPY", "SPDR S&P 500 ETF", "ETF"),
        ("AMD", "Advanced Micro Devices", "Technology"),
    ]
    
    for symbol, name, sector in symbols_data:
        signal = generate_mock_signal()
        signal["symbol"] = symbol
        
        opportunities.append({
            "id": signal["id"],
            "symbol": symbol,
            "name": name,
            "type": "CALL" if signal["type"] == "BUY" else "PUT",
            "confidence": signal["confidence"],
            "potentialReturn": round(random.uniform(5, 20), 1),
            "timeframe": signal["timeframe"],
            "keyReason": f"{signal['pattern']} pattern detected",
            "momentum": random.choice(["strong", "moderate", "building"]),
            "aiScore": round(random.uniform(75, 95), 0),
            "sector": sector,
            "volume": random.randint(10000000, 150000000),
            "volatility": round(random.uniform(0.15, 0.65), 2),
            "currentPrice": round(random.uniform(100, 500), 2),
        })
    
    return {"opportunities": opportunities, "timestamp": now_utc().isoformat()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection_established",
            "timestamp": now_utc().isoformat()
        })
        
        # Create tasks for sending updates
        async def send_market_updates():
            """Send market data updates every 5 seconds"""
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]
            while True:
                try:
                    for symbol in symbols:
                        data = await get_market_data(symbol)
                        await websocket.send_json({
                            "type": "market_update",
                            "data": data
                        })
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Error sending market updates: {e}")
                    break
        
        async def send_signals():
            """Send new signals every 30 seconds"""
            while True:
                try:
                    signal = generate_mock_signal()
                    await websocket.send_json({
                        "type": "new_signal",
                        "data": signal
                    })
                    await asyncio.sleep(30)
                except Exception as e:
                    logger.error(f"Error sending signals: {e}")
                    break
        
        # Run both tasks concurrently
        await asyncio.gather(
            send_market_updates(),
            send_signals(),
            websocket.receive_text()  # Keep connection alive
        )
        
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    print("Starting GoldenSignalsAI Simple Backend")
    print("API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000) 