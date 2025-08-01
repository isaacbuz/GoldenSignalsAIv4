"""
Simple mock server for development
"""
import asyncio
import json
import random
from datetime import datetime, timedelta

import pytz
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mock data
@app.get("/api/v1/market-data/status/market")
async def get_market_status():
    """Mock market status endpoint"""
    now = datetime.now()
    market_hours = 9 <= now.hour < 16  # Simplified market hours

    return {
        "is_open": market_hours,
        "status": "OPEN" if market_hours else "CLOSED",
        "next_open": "09:30 EST" if not market_hours else None,
        "next_close": "16:00 EST" if market_hours else None,
        "timezone": "EST",
        "current_time": now.isoformat(),
    }


@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Mock market data endpoint"""
    base_price = {"SPY": 458, "QQQ": 388, "AAPL": 185}.get(symbol, 100)

    bars = []
    now = int(datetime.now().timestamp())
    for i in range(100):
        price = base_price + random.uniform(-5, 5)
        # Ensure unique timestamps by using integer seconds with proper spacing
        timestamp = now - ((99 - i) * 300)  # 5-minute intervals

        open_price = price - random.uniform(0, 2)
        close_price = price + random.uniform(-1, 1)
        high_price = max(open_price, close_price) + random.uniform(0, 1)
        low_price = min(open_price, close_price) - random.uniform(0, 1)

        bars.append(
            {
                "time": timestamp,  # Use consistent field names
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": random.randint(1000000, 5000000),
            }
        )

    # Ensure bars are sorted by timestamp (ascending order)
    bars.sort(key=lambda x: x["time"])

    return {"bars": bars}


@app.get("/api/v1/signals")
async def get_signals():
    """Mock signals endpoint"""
    signals = []
    symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]

    # Generate signals with unique timestamps spread over the last 24 hours
    now = datetime.now()

    for i in range(5):
        # Spread signals over different time periods
        hours_ago = i * 4  # Signals every 4 hours
        signal_time = now.timestamp() - (hours_ago * 3600)

        signal = {
            "id": f"signal_{i}_{int(signal_time)}",
            "symbol": random.choice(symbols),
            "type": random.choice(["BUY", "SELL"]),
            "confidence": random.randint(70, 95),
            "entry_price": round(random.uniform(100, 500), 2),
            "stop_loss": round(random.uniform(95, 495), 2),
            "take_profit": round(random.uniform(105, 505), 2),
            "timestamp": datetime.fromtimestamp(signal_time).isoformat(),
            "algorithm": "AI_PROPHET_V3",
            "reasoning": "Strong momentum detected with bullish pattern formation",
        }
        signals.append(signal)

    # Sort by timestamp (most recent first)
    signals.sort(key=lambda x: x["timestamp"], reverse=True)

    return signals


@app.get("/api/v1/search/symbols")
async def search_symbols(q: str):
    """Mock symbol search"""
    all_symbols = [
        {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
        {"symbol": "MSFT", "name": "Microsoft Corp.", "exchange": "NASDAQ"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ"},
    ]

    results = [s for s in all_symbols if q.upper() in s["symbol"]]
    return results[:5]


@app.get("/api/v1/market/status")
async def market_status():
    """Mock market status"""
    now = datetime.now()
    is_open = 9 <= now.hour < 16 and now.weekday() < 5

    return {
        "isOpen": is_open,
        "currentTime": now.isoformat(),
        "nextOpen": "2024-01-02T09:30:00",
        "nextClose": "2024-01-02T16:00:00",
    }


@app.get("/api/v1/market-data/quote/{symbol}")
async def get_quote(symbol: str):
    """Mock quote data"""
    base_price = {"SPY": 458, "QQQ": 388, "AAPL": 185}.get(symbol, 100)

    return {
        "symbol": symbol,
        "price": base_price + random.uniform(-2, 2),
        "change": random.uniform(-5, 5),
        "changePercent": random.uniform(-2, 2),
        "volume": random.randint(10000000, 50000000),
        "bid": base_price - 0.01,
        "ask": base_price + 0.01,
        "dayHigh": base_price + 3,
        "dayLow": base_price - 3,
    }


@app.post("/api/logs/frontend")
async def log_frontend_error(request: Request):
    """Mock frontend logging endpoint"""
    try:
        body = await request.json()
        print(f"Frontend Log: {body}")
        return {"status": "logged"}
    except Exception as e:
        print(f"Frontend Log Error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/v1/ai/insights/{symbol}")
async def get_ai_insights(symbol: str):
    """Mock AI insights endpoint"""
    insights = [
        {
            "id": f"insight_{int(datetime.now().timestamp())}",
            "type": "bullish",
            "title": f"Strong momentum detected for {symbol}",
            "description": f"Technical indicators suggest {symbol} is showing strong bullish momentum with RSI at 65 and MACD crossing above signal line.",
            "confidence": random.randint(75, 95),
            "timestamp": datetime.now().isoformat(),
            "priority": "high",
            "tags": [symbol, "momentum", "bullish"],
        },
        {
            "id": f"insight_{int(datetime.now().timestamp()) + 1}",
            "type": "warning",
            "title": "Volume spike observed",
            "description": f"Unusual volume activity detected for {symbol}. Current volume is 3x the daily average.",
            "confidence": random.randint(80, 90),
            "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
            "priority": "medium",
            "tags": [symbol, "volume", "alert"],
        },
    ]
    return {"insights": insights}


@app.get("/api/v1/market/news")
async def get_market_news():
    """Mock market news endpoint"""
    news_items = []

    # Generate mock news items
    news_templates = [
        {
            "title": "Federal Reserve Maintains Interest Rates at Current Level",
            "source": "Reuters",
            "impact": "HIGH",
        },
        {
            "title": "Tech Stocks Rally on Strong Earnings Reports",
            "source": "CNBC",
            "impact": "MEDIUM",
        },
        {
            "title": "Oil Prices Rise Amid Supply Chain Concerns",
            "source": "Bloomberg",
            "impact": "MEDIUM",
        },
        {
            "title": "Consumer Confidence Index Beats Expectations",
            "source": "MarketWatch",
            "impact": "LOW",
        },
        {
            "title": "Cryptocurrency Market Shows Signs of Recovery",
            "source": "CoinDesk",
            "impact": "LOW",
        },
    ]

    for i, template in enumerate(news_templates):
        news_time = datetime.now() - timedelta(hours=i * 2)
        news_items.append(
            {
                "id": f"news_{i}_{int(news_time.timestamp())}",
                "title": template["title"],
                "source": template["source"],
                "impact": template["impact"],
                "timestamp": news_time.isoformat(),
                "url": f"https://example.com/news/{i}",
                "summary": f"This is a summary of the news article: {template['title']}",
            }
        )

    return news_items


@app.get("/api/v1/market/trending")
async def get_trending_symbols():
    """Mock trending symbols endpoint"""
    trending_symbols = [
        {
            "symbol": "SPY",
            "name": "SPDR S&P 500 ETF Trust",
            "type": "ETF",
            "exchange": "NYSE",
            "currency": "USD",
        },
        {
            "symbol": "QQQ",
            "name": "Invesco QQQ Trust",
            "type": "ETF",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "type": "Stock",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "MSFT",
            "name": "Microsoft Corporation",
            "type": "Stock",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc.",
            "type": "Stock",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "AMZN",
            "name": "Amazon.com Inc.",
            "type": "Stock",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "TSLA",
            "name": "Tesla Inc.",
            "type": "Stock",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "NVDA",
            "name": "NVIDIA Corporation",
            "type": "Stock",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "META",
            "name": "Meta Platforms Inc.",
            "type": "Stock",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "NFLX",
            "name": "Netflix Inc.",
            "type": "Stock",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
    ]

    return {"symbols": trending_symbols}


@app.get("/api/v1/market-data/{symbol}/historical")
async def get_historical_data(symbol: str, timeframe: str = "15m", limit: int = 500):
    """Mock historical data endpoint"""
    base_price = {"SPY": 458, "QQQ": 388, "AAPL": 185}.get(symbol, 100)

    bars = []
    now = datetime.now().timestamp()

    # Generate historical data based on timeframe
    interval_seconds = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1D": 86400,
    }.get(timeframe, 900)

    for i in range(min(limit, 500)):
        timestamp = int(now - ((limit - i - 1) * interval_seconds))  # Ensure integer timestamps
        price = base_price + random.uniform(-10, 10)

        open_price = price - random.uniform(0, 2)
        close_price = price + random.uniform(-1, 1)
        high_price = max(open_price, close_price) + random.uniform(0, 2)
        low_price = min(open_price, close_price) - random.uniform(0, 2)

        bars.append(
            {
                "time": timestamp,  # Use 'time' instead of 't' for consistency
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": random.randint(1000000, 5000000),
            }
        )

    return {"bars": bars}


# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    signal_counter = 0
    try:
        while True:
            # Send mock real-time data with unique timestamps
            signal_counter += 1
            current_time = datetime.now()

            data = {
                "type": "signal",
                "data": {
                    "id": f"ws_signal_{signal_counter}_{int(current_time.timestamp())}",
                    "symbol": random.choice(["SPY", "QQQ", "AAPL"]),
                    "type": random.choice(["BUY", "SELL"]),
                    "confidence": random.randint(70, 95),
                    "price": round(random.uniform(100, 500), 2),
                    "timestamp": current_time.isoformat(),
                },
            }
            await websocket.send_json(data)
            await asyncio.sleep(30)  # Send signals every 30 seconds instead of 5
    except Exception:
        pass


if __name__ == "__main__":
    print("🚀 Starting Mock Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
