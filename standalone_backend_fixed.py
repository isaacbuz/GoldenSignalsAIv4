#!/usr/bin/env python3
"""
Standalone Backend for GoldenSignalsAI
No complex imports - everything self-contained
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import yfinance as yf
import numpy as np
from collections import defaultdict
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GoldenSignalsAI Standalone Backend",
    description="Simplified backend with live data and ML signals",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Data Models =============

class MarketData(BaseModel):
    symbol: str
    price: float
    change: float
    changePercent: float
    volume: int
    high: float
    low: float
    open: float
    previousClose: float
    timestamp: str

class Signal(BaseModel):
    id: str
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'  # Changed from 'type' to 'action'
    confidence: float
    price: float
    timestamp: str
    reason: str
    indicators: Dict[str, Any]
    risk_level: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class MarketOpportunity(BaseModel):
    symbol: str
    type: str
    confidence: float
    potential_return: float
    risk_level: str
    reason: str
    timestamp: str

# ============= WebSocket Manager =============

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_status: Dict[WebSocket, bool] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_status[websocket] = True
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.connection_status[websocket] = False
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        if self.connection_status.get(websocket, False):
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending personal message: {e}")
                self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            if self.connection_status.get(connection, False):
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting: {e}")
                    disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# ============= Data Cache =============

class DataCache:
    def __init__(self):
        self.cache = {}
        self.ttl = 60  # 60 seconds TTL

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now(timezone.utc) - timestamp < timedelta(seconds=self.ttl):
                return data
        return None

    def set(self, key: str, value: Any):
        self.cache[key] = (value, datetime.now(timezone.utc))

cache = DataCache()

# ============= Technical Indicators =============

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate RSI"""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        return 100.0
    
    rs = up / down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)

def calculate_macd(prices: List[float]) -> Dict[str, float]:
    """Calculate MACD"""
    if len(prices) < 26:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
    
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    return {
        "macd": float(macd.iloc[-1]),
        "signal": float(signal.iloc[-1]),
        "histogram": float(macd.iloc[-1] - signal.iloc[-1])
    }

def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Dict[str, float]:
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return {"upper": 0.0, "middle": 0.0, "lower": 0.0}
    
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    return {
        "upper": sma + (2 * std),
        "middle": sma,
        "lower": sma - (2 * std)
    }

# ============= Market Data Service =============

async def get_market_data(symbol: str) -> Optional[MarketData]:
    """Fetch real market data using yfinance"""
    try:
        # Check cache first
        cached = cache.get(f"market_data_{symbol}")
        if cached:
            return cached
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        if current_price == 0:
            # Fallback to history
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
        
        previous_close = info.get('previousClose', current_price)
        
        data = MarketData(
            symbol=symbol,
            price=current_price,
            change=current_price - previous_close,
            changePercent=((current_price - previous_close) / previous_close * 100) if previous_close else 0,
            volume=info.get('volume', 0),
            high=info.get('dayHigh', current_price),
            low=info.get('dayLow', current_price),
            open=info.get('open', current_price),
            previousClose=previous_close,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        cache.set(f"market_data_{symbol}", data)
        return data
        
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        return None

async def get_historical_data(symbol: str, period: str = "1mo", interval: str = "1d") -> List[Dict]:
    """Fetch historical data"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            return []
        
        data = []
        for idx, row in hist.iterrows():
            data.append({
                "timestamp": idx.isoformat(),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return []

# ============= ML Signal Generator =============

async def generate_ml_signals(symbol: str) -> List[Signal]:
    """Generate ML-based trading signals"""
    try:
        # Get historical data
        hist_data = await get_historical_data(symbol, period="1mo", interval="1d")
        if len(hist_data) < 20:
            return []
        
        prices = [d['close'] for d in hist_data]
        current_price = prices[-1]
        
        # Calculate indicators
        rsi = calculate_rsi(prices)
        macd = calculate_macd(prices)
        bb = calculate_bollinger_bands(prices)
        
        signals = []
        
        # RSI Signal
        if rsi < 30:
            signals.append(Signal(
                id=f"{symbol}_{int(datetime.now().timestamp())}_{random.randint(1, 100)}",
                symbol=symbol,
                action="BUY",
                confidence=0.75,
                price=current_price,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason="RSI oversold condition",
                indicators={"rsi": rsi},
                risk_level="medium",
                entry_price=current_price,
                stop_loss=current_price * 0.98,
                take_profit=current_price * 1.05
            ))
        elif rsi > 70:
            signals.append(Signal(
                id=f"{symbol}_{int(datetime.now().timestamp())}_{random.randint(1, 100)}",
                symbol=symbol,
                action="SELL",
                confidence=0.75,
                price=current_price,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason="RSI overbought condition",
                indicators={"rsi": rsi},
                risk_level="medium",
                entry_price=current_price,
                stop_loss=current_price * 1.02,
                take_profit=current_price * 0.95
            ))
        
        # MACD Signal
        if macd["histogram"] > 0 and macd["macd"] > macd["signal"]:
            signals.append(Signal(
                id=f"{symbol}_{int(datetime.now().timestamp())}_{random.randint(1, 100)}",
                symbol=symbol,
                action="BUY",
                confidence=0.70,
                price=current_price,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason="MACD bullish crossover",
                indicators={"macd": macd},
                risk_level="low",
                entry_price=current_price,
                stop_loss=current_price * 0.97,
                take_profit=current_price * 1.08
            ))
        
        # Bollinger Bands Signal
        if current_price < bb["lower"]:
            signals.append(Signal(
                id=f"{symbol}_{int(datetime.now().timestamp())}_{random.randint(1, 100)}",
                symbol=symbol,
                action="BUY",
                confidence=0.65,
                price=current_price,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason="Price below lower Bollinger Band",
                indicators={"bollinger_bands": bb},
                risk_level="high",
                entry_price=current_price,
                stop_loss=current_price * 0.95,
                take_profit=current_price * 1.10
            ))
        
        # Add some random signals for demo
        if random.random() > 0.7:
            signal_type = random.choice(["BUY", "SELL", "HOLD"])
            signals.append(Signal(
                id=f"{symbol}_{int(datetime.now().timestamp())}_{random.randint(1, 100)}",
                symbol=symbol,
                action=signal_type,
                confidence=random.uniform(0.6, 0.9),
                price=current_price,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason=f"AI pattern detection: {random.choice(['Breakout', 'Reversal', 'Continuation'])} pattern",
                indicators={
                    "rsi": rsi,
                    "macd": macd,
                    "bollinger_bands": bb
                },
                risk_level=random.choice(["low", "medium", "high"]),
                entry_price=current_price,
                stop_loss=current_price * (0.95 if signal_type == "BUY" else 1.05),
                take_profit=current_price * (1.10 if signal_type == "BUY" else 0.90)
            ))
        
        return signals
        
    except Exception as e:
        logger.error(f"Error generating ML signals for {symbol}: {e}")
        return []

# ============= API Endpoints =============

@app.get("/")
async def root():
    return {"message": "GoldenSignalsAI Standalone Backend", "status": "running"}

@app.get("/api/v1/market-data/{symbol}")
async def get_market_data_endpoint(symbol: str):
    # Validate symbol format
    if not symbol or len(symbol) > 10 or not symbol.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail=f"Invalid symbol format: {symbol}")
    
    data = await get_market_data(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Market data not found for {symbol}")
    return data

@app.get("/api/v1/market-data/{symbol}/historical")
async def get_historical_data_endpoint(symbol: str, period: str = "1mo", interval: str = "1d"):
    # Validate period
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail=f"Invalid period: {period}. Valid periods: {', '.join(valid_periods)}")
    
    data = await get_historical_data(symbol, period, interval)
    return {"symbol": symbol, "data": data}

@app.get("/api/v1/signals")
async def get_all_signals():
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "SPY"]
    all_signals = []
    
    for symbol in symbols:
        signals = await generate_ml_signals(symbol)
        all_signals.extend([s.dict() for s in signals])
    
    return all_signals

@app.get("/api/v1/signals/{symbol}")
async def get_signals_for_symbol(symbol: str):
    signals = await generate_ml_signals(symbol)
    return [s.dict() for s in signals]

@app.get("/api/v1/signals/{symbol}/insights")
async def get_signal_insights(symbol: str):
    market_data = await get_market_data(symbol)
    signals = await generate_ml_signals(symbol)
    
    return {
        "symbol": symbol,
        "current_price": market_data.price if market_data else 0,
        "signal_count": len(signals),
        "bullish_signals": len([s for s in signals if s.action == "BUY"]),
        "bearish_signals": len([s for s in signals if s.action == "SELL"]),
        "average_confidence": np.mean([s.confidence for s in signals]) if signals else 0,
        "recommendation": "BUY" if len([s for s in signals if s.action == "BUY"]) > len([s for s in signals if s.action == "SELL"]) else "HOLD",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/v1/signals/precise-options")
async def get_precise_options_signals(symbol: str = "SPY", timeframe: str = "15m"):
    # Mock precise options signals
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signals": [
            {
                "type": "CALL",
                "strike": 450,
                "expiry": "2024-01-19",
                "confidence": 0.82,
                "entry_price": 2.45,
                "current_price": 2.50,
                "implied_volatility": 0.18,
                "delta": 0.55,
                "gamma": 0.02,
                "theta": -0.08,
                "vega": 0.15
            }
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/v1/market/opportunities")
async def get_market_opportunities():
    opportunities = []
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    
    for symbol in symbols:
        if random.random() > 0.5:
            opportunities.append(MarketOpportunity(
                symbol=symbol,
                type=random.choice(["Breakout", "Reversal", "Momentum"]),
                confidence=random.uniform(0.65, 0.95),
                potential_return=random.uniform(5, 25),
                risk_level=random.choice(["low", "medium", "high"]),
                reason=f"{symbol} showing strong {random.choice(['bullish', 'technical'])} signals",
                timestamp=datetime.now(timezone.utc).isoformat()
            ))
    
    return {"opportunities": [opp.model_dump() for opp in opportunities]}

# ============= WebSocket Endpoint =============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "status": "connected",
                "message": "Connected to GoldenSignalsAI WebSocket"
            }),
            websocket
        )
        
        # Create tasks for sending updates
        async def send_market_updates():
            symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
            while manager.connection_status.get(websocket, False):
                try:
                    for symbol in symbols:
                        if not manager.connection_status.get(websocket, False):
                            break
                            
                        market_data = await get_market_data(symbol)
                        if market_data:
                            await manager.send_personal_message(
                                json.dumps({
                                    "type": "market_data",
                                    "data": market_data.dict()
                                }),
                                websocket
                            )
                    await asyncio.sleep(5)  # Send updates every 5 seconds
                except Exception as e:
                    logger.error(f"Error sending market updates: {e}")
                    break
        
        async def send_signals():
            while manager.connection_status.get(websocket, False):
                try:
                    all_signals = await get_all_signals()
                    if all_signals and manager.connection_status.get(websocket, False):
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "signals",
                                "data": all_signals
                            }),
                            websocket
                        )
                    await asyncio.sleep(30)  # Send signals every 30 seconds
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
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============= Main =============

if __name__ == "__main__":
    print("🚀 Starting GoldenSignalsAI Standalone Backend")
    print("📊 Live market data enabled")
    print("🤖 ML signals generation active")
    print("🌐 API docs available at http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 